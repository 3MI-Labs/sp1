//! # The `program` Module
//!
//! This module defines the `ProgramChip` type whose trace counts the multiplicities of instruction calls inside a
//! [`Program`].

use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use std::collections::HashMap;

use crate::{air::ProgramAirBuilder, utils::pad_rows_fixed};
use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::PrimeField;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use sp1_core_executor::{ExecutionRecord, Program};
use sp1_derive::AlignedBorrow;
use sp1_stark::air::{MachineAir, SP1AirBuilder};

use crate::cpu::columns::{InstructionCols, OpcodeSelectorCols};

/// The number of preprocessed program columns.
pub const NUM_PROGRAM_PREPROCESSED_COLS: usize = size_of::<ProgramPreprocessedCols<u8>>();

/// The number of columns for the program multiplicities.
pub const NUM_PROGRAM_MULT_COLS: usize = size_of::<ProgramMultiplicityCols<u8>>();

/// The virtual columns of the preprocessed trace for the [ProgramChip].
///
/// Since the program is known before prove-time, the virtual trace of the [`ProgramChip`] these columns hold the
/// information on which instruction is used for each value of the program counter.
#[derive(AlignedBorrow, Clone, Copy, Default)]
#[repr(C)]
pub struct ProgramPreprocessedCols<T> {
    /// The value of the program counter.
    pub pc: T,
    /// The information on the instruction to be performed at this program counter value.
    pub instruction: InstructionCols<T>,
    /// The opcode selectors for this instruction
    pub selectors: OpcodeSelectorCols<T>,
}

/// The real columns of the trace for the [ProgramChip], recording the shard index in which an instruction is called and
/// its multiplicity.
#[derive(AlignedBorrow, Clone, Copy, Default)]
#[repr(C)]
pub struct ProgramMultiplicityCols<T> {
    pub shard: T,
    pub multiplicity: T,
}

/// The chip which handles the execution of the program.
///
/// - Virtual trace: populated from a [`Program`] object by appending a row for every
///   [`Instruction`](sp1_core_executor::Instruction).
/// - Dependencies: none.
/// - Real trace: populated from an [`ExecutionRecord`] object by counting the multiplicity of each
///   [`Instruction`](sp1_core_executor::Instruction) in the record and appending a trace row (containing the shard
///   index and the multiplicity value) for each [`Instruction`](sp1_core_executor::Instruction).
#[derive(Default)]
pub struct ProgramChip;

impl ProgramChip {
    pub const fn new() -> Self {
        Self {}
    }
}

impl<F: PrimeField> MachineAir<F> for ProgramChip {
    type Record = ExecutionRecord;

    type Program = Program;

    /// Gives this chip's name as `"Program"`.
    fn name(&self) -> String {
        "Program".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_PROGRAM_PREPROCESSED_COLS
    }

    /// Generates the preprocessed trace of the program chip for the given [`Program`] by populating a row of
    /// [`ProgramPreprocessedCols`] for each [`Instruction`](sp1_core_executor::Instruction) in the [`Program`].
    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<RowMajorMatrix<F>> {
        debug_assert!(
            !program.instructions.is_empty() || program.preprocessed_shape.is_some(),
            "empty program"
        );

        let mut preprocessed_rows: Vec<[F; 37]> = program
            // Enumerate over the instructions of the given program
            .instructions
            .iter()
            .enumerate()
            // Map each instruction to a populated row for the preprocessed trace and collect in a Vec
            .map(|(i, &instruction)| {
                // ... starting from base, increment by 4 to maintain alignment,
                let pc = program.pc_base + (i as u32 * 4);
                // ... initialise an empty row
                let mut row = [F::zero(); NUM_PROGRAM_PREPROCESSED_COLS];
                // ... populate the row with the program counter and the corresponding instruction
                let cols: &mut ProgramPreprocessedCols<F> = row.as_mut_slice().borrow_mut();
                cols.pc = F::from_canonical_u32(pc);
                cols.instruction.populate(instruction);
                cols.selectors.populate(instruction);

                row
            })
            .collect();

        // Pad the trace to a power of two with all-zero rows depending on the proof shape in `program`.
        pad_rows_fixed(
            &mut preprocessed_rows,
            || [F::zero(); NUM_PROGRAM_PREPROCESSED_COLS],
            program.fixed_log2_rows::<F, _>(self),
        );

        // Convert the Vec of rows to a row-major trace matrix
        let preprocessed_trace = RowMajorMatrix::new(
            preprocessed_rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_PROGRAM_PREPROCESSED_COLS,
        );

        Some(preprocessed_trace)
    }

    /// A [`ProgramChip`] doesn't depend on other chips, so this doesn't do anything.
    fn generate_dependencies(&self, _input: &ExecutionRecord, _output: &mut ExecutionRecord) {
        // Do nothing since this chip has no dependencies.
    }

    /// Generates the trace of the program chip for the given [`ExecutionRecord`] by populating a row of
    /// [`ProgramMultiplicityCols`] for each [`Instruction`](sp1_core_executor::Instruction) in the
    /// [`ExecutionRecord::program`] pointed to by `input`.
    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        // Count the number of times each instruction is called from the list of CPU events.
        // Store it as a map of PC -> count.
        let mut instruction_counts = HashMap::new();
        input.cpu_events.iter().for_each(|event| {
            let pc = event.pc;
            instruction_counts.entry(pc).and_modify(|count| *count += 1).or_insert(1);
        });

        let mut trace_rows: Vec<[F; 2]> = input
            // Enumerate over the instructions in the input record's program.
            .program
            .instructions
            .clone()
            .into_iter()
            .enumerate()
            // Populate the trace row with the multiplicity of the instruction based on the count done above, or with 0
            // if the instruction isn't called by the CPU events.
            .map(|(i, _)| {
                let pc = input.program.pc_base + (i as u32 * 4);
                let mut row = [F::zero(); NUM_PROGRAM_MULT_COLS];
                let cols: &mut ProgramMultiplicityCols<F> = row.as_mut_slice().borrow_mut();
                cols.shard = F::from_canonical_u32(input.public_values.execution_shard);
                cols.multiplicity =
                    F::from_canonical_usize(*instruction_counts.get(&pc).unwrap_or(&0));
                row
            })
            .collect();

        // Pad the trace to a power of two depending on the proof shape in `input`.
        pad_rows_fixed(
            &mut trace_rows,
            || [F::zero(); NUM_PROGRAM_MULT_COLS],
            input.fixed_log2_rows::<F, _>(self),
        );

        // Convert the Vec of trace rows to a row-major trace matrix and return.
        RowMajorMatrix::new(
            trace_rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_PROGRAM_MULT_COLS,
        )
    }

    fn included(&self, _: &Self::Record) -> bool {
        true
    }
}

impl<F> BaseAir<F> for ProgramChip {
    /// The number of multiplicity columns for a [`ProgramChip`].
    fn width(&self) -> usize {
        NUM_PROGRAM_MULT_COLS
    }
}

impl<AB> Air<AB> for ProgramChip
where
    AB: SP1AirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let preprocessed = builder.preprocessed();

        let prep_local = preprocessed.row_slice(0);
        let prep_local: &ProgramPreprocessedCols<AB::Var> = (*prep_local).borrow();
        let mult_local = main.row_slice(0);
        let mult_local: &ProgramMultiplicityCols<AB::Var> = (*mult_local).borrow();

        // Constrain the interaction with CPU table
        builder.receive_program(
            prep_local.pc,
            prep_local.instruction,
            prep_local.selectors,
            mult_local.shard,
            mult_local.multiplicity,
        );
    }
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use hashbrown::HashMap;
    use p3_baby_bear::BabyBear;

    use p3_matrix::dense::RowMajorMatrix;
    use sp1_core_executor::{ExecutionRecord, Instruction, Opcode, Program};
    use sp1_stark::air::MachineAir;

    use crate::program::ProgramChip;

    #[test]
    fn generate_trace() {
        // main:
        //     addi x29, x0, 5
        //     addi x30, x0, 37
        //     add x31, x30, x29
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 37, false, true),
            Instruction::new(Opcode::ADD, 31, 30, 29, false, false),
        ];
        let shard = ExecutionRecord {
            program: Arc::new(Program {
                instructions,
                pc_start: 0,
                pc_base: 0,
                memory_image: HashMap::new(),
                preprocessed_shape: None,
            }),
            ..Default::default()
        };
        let chip = ProgramChip::new();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        println!("{:?}", trace.values)
    }
}
