use std::borrow::BorrowMut;

use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use sp1_core_executor::{ByteOpcode, ExecutionRecord, Program};
use sp1_stark::air::MachineAir;

use crate::utils::zeroed_f_vec;

use super::{
    columns::{ByteMultCols, NUM_BYTE_MULT_COLS, NUM_BYTE_PREPROCESSED_COLS},
    ByteChip,
};

pub const NUM_ROWS: usize = 1 << 16;

impl<F: Field> MachineAir<F> for ByteChip<F> {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Byte".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_BYTE_PREPROCESSED_COLS
    }

    fn generate_preprocessed_trace(&self, _program: &Self::Program) -> Option<RowMajorMatrix<F>> {
        let trace = Self::trace();
        Some(trace)
    }

    fn generate_dependencies(&self, _input: &ExecutionRecord, _output: &mut ExecutionRecord) {
        // Do nothing since this chip has no dependencies.
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        // Generate a matrix of 2^16 rows where the row index are the operands of the byte operations and the cells
        // contain the multiplicities of each byte operation for those two operands.
        //
        // The length of the trace therefore doesn't depend on the number of lookup events.
        let mut trace =
            RowMajorMatrix::new(zeroed_f_vec(NUM_BYTE_MULT_COLS * NUM_ROWS), NUM_BYTE_MULT_COLS);

        // For each shard ...
        for (_shard_index, shard_byte_lookup_events) in input.byte_lookups.iter() {
            // ... iterate over the different lookup events.
            for (lookup_event, mult) in shard_byte_lookup_events.iter() {
                // If the lookup is not a 16-bit range check ...
                let row_index = if lookup_event.opcode != ByteOpcode::U16Range {
                    // ... compute the row index as the concatenation of the `b` and `c` operands.
                    (((lookup_event.b as u16) << 8) + lookup_event.c as u16) as usize
                } else {
                    // Otherwise, the row index is the `a1` operand
                    lookup_event.a1 as usize
                };

                let col_index = lookup_event.opcode as usize;

                // Get a reference to the corresponding row, cast is as a ByteMultCols, and increment the multiplicity
                // corresponding to the OpCode.
                let cols: &mut ByteMultCols<F> = trace.row_mut(row_index).borrow_mut();
                cols.multiplicities[col_index] += F::from_canonical_usize(*mult);
            }
        }

        trace
    }

    fn included(&self, _shard: &Self::Record) -> bool {
        true
    }
}
