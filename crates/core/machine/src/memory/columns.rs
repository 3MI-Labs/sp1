//! Several types to hold the columns relevant to each memory access.
//!
//! The core type is [MemoryAccessCols] which contains the access's information (value, timestamp, etc.). Three wrapper
//! types represent the possible kinds of access:
//!
//! - Read
//! - Write
//! - Read-Write
//!
//! Also, the [MemoryCols] trait, implemented for the different memory columns types, which defines function signatures
//! to access the fields common across the four access types.

use sp1_derive::AlignedBorrow;
use sp1_stark::Word;

/// The core memory access columns; this doesn't look like it should be accessed directly, but instead should access
/// through the [MemoryCols] trait functions.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryAccessCols<T> {
    /// The value of the memory access.
    pub value: Word<T>,

    /// The previous shard and timestamp that this memory access is being read from.
    pub prev_shard: T,
    pub prev_clk: T,

    /// This will be true if the current shard == prev_access's shard, else false.
    pub compare_clk: T,

    /// The following columns are decomposed limbs for the difference between the current access's
    /// timestamp and the previous access's timestamp.  Note the actual value of the timestamp
    /// is either the accesses' shard or clk depending on the value of compare_clk.

    /// This column is the least significant 16 bit limb of current access timestamp - prev access
    /// timestamp.
    pub diff_16bit_limb: T,

    /// This column is the most significant 8 bit limb of current access timestamp - prev access
    /// timestamp.
    pub diff_8bit_limb: T,
}

/// Memory read access; it contains only an inner [MemoryAccessCols] field.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryReadCols<T> {
    pub access: MemoryAccessCols<T>,
}

/// Memory write access; it contains an inner [MemoryAccessCols] field, as well as the previous value that was
/// overwritten by this write access.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryWriteCols<T> {
    pub prev_value: Word<T>,
    pub access: MemoryAccessCols<T>,
}

/// Memory read-write access; it contains an inner [MemoryAccessCols] field, as well as the previous value that was
/// overwritten by this write access after being read.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryReadWriteCols<T> {
    pub prev_value: Word<T>,
    pub access: MemoryAccessCols<T>,
}

/// The common columns for all memory access types.
pub trait MemoryCols<T> {
    fn access(&self) -> &MemoryAccessCols<T>;

    fn access_mut(&mut self) -> &mut MemoryAccessCols<T>;

    fn prev_value(&self) -> &Word<T>;

    fn prev_value_mut(&mut self) -> &mut Word<T>;

    fn value(&self) -> &Word<T>;

    fn value_mut(&mut self) -> &mut Word<T>;
}

impl<T> MemoryCols<T> for MemoryReadCols<T> {
    fn access(&self) -> &MemoryAccessCols<T> {
        &self.access
    }

    fn access_mut(&mut self) -> &mut MemoryAccessCols<T> {
        &mut self.access
    }

    fn prev_value(&self) -> &Word<T> {
        &self.access.value
    }

    fn prev_value_mut(&mut self) -> &mut Word<T> {
        &mut self.access.value
    }

    fn value(&self) -> &Word<T> {
        &self.access.value
    }

    fn value_mut(&mut self) -> &mut Word<T> {
        &mut self.access.value
    }
}

impl<T> MemoryCols<T> for MemoryWriteCols<T> {
    fn access(&self) -> &MemoryAccessCols<T> {
        &self.access
    }

    fn access_mut(&mut self) -> &mut MemoryAccessCols<T> {
        &mut self.access
    }

    fn prev_value(&self) -> &Word<T> {
        &self.prev_value
    }

    fn prev_value_mut(&mut self) -> &mut Word<T> {
        &mut self.prev_value
    }

    fn value(&self) -> &Word<T> {
        &self.access.value
    }

    fn value_mut(&mut self) -> &mut Word<T> {
        &mut self.access.value
    }
}

impl<T> MemoryCols<T> for MemoryReadWriteCols<T> {
    fn access(&self) -> &MemoryAccessCols<T> {
        &self.access
    }

    fn access_mut(&mut self) -> &mut MemoryAccessCols<T> {
        &mut self.access
    }

    fn prev_value(&self) -> &Word<T> {
        &self.prev_value
    }

    fn prev_value_mut(&mut self) -> &mut Word<T> {
        &mut self.prev_value
    }

    fn value(&self) -> &Word<T> {
        &self.access.value
    }

    fn value_mut(&mut self) -> &mut Word<T> {
        &mut self.access.value
    }
}

/// A utility method to convert a slice of memory access columns into a vector of values.
/// This is useful for comparing the values of a memory access to limbs.
pub fn value_as_limbs<T: Clone, M: MemoryCols<T>>(memory: &[M]) -> Vec<T> {
    memory.iter().flat_map(|m| m.value().clone().into_iter()).collect()
}
