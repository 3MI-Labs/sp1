//! Functions for each kind of memory access columns to populate [MemoryAccessCols] with access value and timestamp
//! information as well as produce [ByteRecord]s with the appropriate U16 and U8 range checks.

use p3_field::PrimeField32;
use sp1_core_executor::events::{
    ByteRecord, MemoryReadRecord, MemoryRecord, MemoryRecordEnum, MemoryWriteRecord,
};

use super::{MemoryAccessCols, MemoryReadCols, MemoryReadWriteCols, MemoryWriteCols};

impl<F: PrimeField32> MemoryWriteCols<F> {
    /// Populate the `output` with the byte look-up events for the given [MemoryWriteRecord]. This also saves the
    /// previous memory access value into `self`.
    pub fn populate(&mut self, record: MemoryWriteRecord, output: &mut impl ByteRecord) {
        // Destructure the given MemoryWriteRecord
        let current_record =
            MemoryRecord { value: record.value, shard: record.shard, timestamp: record.timestamp };
        let prev_record = MemoryRecord {
            value: record.prev_value,
            shard: record.prev_shard,
            timestamp: record.prev_timestamp,
        };

        // Save the previous value into `self`
        self.prev_value = prev_record.value.into();

        // Populate the inner MemoryAccessCols
        self.access.populate_access(current_record, prev_record, output);
    }
}

impl<F: PrimeField32> MemoryReadCols<F> {
    /// Populate the `output` with the byte look-up events for the given [MemoryReadRecord].
    pub fn populate(&mut self, record: MemoryReadRecord, output: &mut impl ByteRecord) {
        // Destructure the given MemoryReadRecord
        let current_record =
            MemoryRecord { value: record.value, shard: record.shard, timestamp: record.timestamp };
        let prev_record = MemoryRecord {
            value: record.value,
            shard: record.prev_shard,
            timestamp: record.prev_timestamp,
        };

        // Populate the inner MemoryAccessCols
        self.access.populate_access(current_record, prev_record, output);
    }
}

impl<F: PrimeField32> MemoryReadWriteCols<F> {
    /// Populate the `output` with the byte look-up events for the given [MemoryRecordEnum] depending on the variant.
    pub fn populate(&mut self, record: MemoryRecordEnum, output: &mut impl ByteRecord) {
        match record {
            MemoryRecordEnum::Read(read_record) => self.populate_read(read_record, output),
            MemoryRecordEnum::Write(write_record) => self.populate_write(write_record, output),
        }
    }

    /// Populate the `output` with the byte look-up events for the given [MemoryWriteRecord]. This also saves the
    /// previous memory access value into `self`.
    pub fn populate_write(&mut self, record: MemoryWriteRecord, output: &mut impl ByteRecord) {
        let current_record =
            MemoryRecord { value: record.value, shard: record.shard, timestamp: record.timestamp };
        let prev_record = MemoryRecord {
            value: record.prev_value,
            shard: record.prev_shard,
            timestamp: record.prev_timestamp,
        };
        self.prev_value = prev_record.value.into();
        self.access.populate_access(current_record, prev_record, output);
    }

    /// Populate the `output` with the byte look-up events for the given [MemoryReadRecord]. This also saves the
    /// previous memory access value into `self`.
    pub fn populate_read(&mut self, record: MemoryReadRecord, output: &mut impl ByteRecord) {
        let current_record =
            MemoryRecord { value: record.value, shard: record.shard, timestamp: record.timestamp };
        let prev_record = MemoryRecord {
            value: record.value,
            shard: record.prev_shard,
            timestamp: record.prev_timestamp,
        };
        self.prev_value = prev_record.value.into();
        self.access.populate_access(current_record, prev_record, output);
    }
}

impl<F: PrimeField32> MemoryAccessCols<F> {
    /// Populate the information of the memory access based on the previous [MemoryRecord] and the current one which
    /// updates it, and write into the `output` the U16 and U8 range checks on the limbs of the diff of the access
    /// timestamps.
    pub(crate) fn populate_access(
        &mut self,
        current_record: MemoryRecord,
        prev_record: MemoryRecord,
        output: &mut impl ByteRecord,
    ) {
        self.value = current_record.value.into();

        self.prev_shard = F::from_canonical_u32(prev_record.shard);
        self.prev_clk = F::from_canonical_u32(prev_record.timestamp);

        // Check whether clock-level comparison is required when the previous access is in the same shard as the
        // current one.
        let use_clk_comparison = prev_record.shard == current_record.shard;
        self.compare_clk = F::from_bool(use_clk_comparison);

        // Conditionally compute the previous and current time values
        let prev_time_value =
            if use_clk_comparison { prev_record.timestamp } else { prev_record.shard };
        let current_time_value =
            if use_clk_comparison { current_record.timestamp } else { current_record.shard };

        // Compute the diff of the time values
        let diff_minus_one = current_time_value - prev_time_value - 1;

        // Split and record the limbs of the diff
        let diff_16bit_limb = (diff_minus_one & 0xffff) as u16;
        self.diff_16bit_limb = F::from_canonical_u16(diff_16bit_limb);
        let diff_8bit_limb = (diff_minus_one >> 16) & 0xff;
        self.diff_8bit_limb = F::from_canonical_u32(diff_8bit_limb);

        // Generate U16 and U8 range checks on the limbs of the diff of timestamps
        output.add_u16_range_check(current_record.shard, diff_16bit_limb);
        output.add_u8_range_check(current_record.shard, 0, diff_8bit_limb as u8);
    }
}
