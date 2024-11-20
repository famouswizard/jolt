use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, SubtableIndices};
use crate::{
    field::JoltField,
    jolt::subtable::{low_bit::LowBitSubtable, LassoSubtable},
    utils::instruction_utils::chunk_operand_usize,
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct AssertAlignedMemoryAccessInstruction<const WORD_SIZE: usize, const ALIGN: usize>(
    pub u64,
);

impl<const WORD_SIZE: usize, const ALIGN: usize> JoltInstruction
    for AssertAlignedMemoryAccessInstruction<WORD_SIZE, ALIGN>
{
    fn operands(&self) -> (u64, u64) {
        (self.0, 0)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], _: usize, _: usize) -> F {
        match ALIGN {
            2 => {
                assert_eq!(vals.len(), 1);
                let lowest_bit = vals[0];
                F::one() - lowest_bit
            }
            4 => {
                assert_eq!(vals.len(), 2);
                let lowest_bit = vals[0];
                let second_lowest_bit = vals[1];
                (F::one() - lowest_bit) * (F::one() - second_lowest_bit)
            }
            _ => panic!("ALIGN must be either 2 (for LH, LHU, SH) or 4 (for LW, SW)"),
        }
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        match ALIGN {
            2 => 1,
            4 => 2,
            _ => panic!("ALIGN must be either 2 (for LH, LHU, SH) or 4 (for LW, SW)"),
        }
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        match ALIGN {
            2 => vec![(
                Box::new(LowBitSubtable::<F, 0>::new()),
                SubtableIndices::from(C - 1),
            )],
            4 => vec![
                (
                    Box::new(LowBitSubtable::<F, 0>::new()),
                    SubtableIndices::from(C - 1),
                ),
                (
                    Box::new(LowBitSubtable::<F, 1>::new()),
                    SubtableIndices::from(C - 1),
                ),
            ],
            _ => panic!("ALIGN must be either 2 (for LH, LHU, SH) or 4 (for LW, SW)"),
        }
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_operand_usize(self.0, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        (self.0 % ALIGN as u64 == 0) as u64
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        if WORD_SIZE == 32 {
            Self(rng.next_u32() as u64)
        } else if WORD_SIZE == 64 {
            Self(rng.next_u64())
        } else {
            panic!("Only 32-bit and 64-bit word sizes are supported");
        }
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    use super::AssertAlignedMemoryAccessInstruction;

    #[test]
    fn assert_aligned_memory_access_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // ALIGN = 2
        for _ in 0..256 {
            let x = rng.next_u64();
            let instruction = AssertAlignedMemoryAccessInstruction::<WORD_SIZE, 2>(x);
            jolt_instruction_test!(instruction);
        }
        // ALIGN = 4
        for _ in 0..256 {
            let x = rng.next_u64();
            let instruction = AssertAlignedMemoryAccessInstruction::<WORD_SIZE, 4>(x);
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn assert_aligned_memory_access_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // ALIGN = 2
        for _ in 0..256 {
            let x = rng.next_u64();
            let instruction = AssertAlignedMemoryAccessInstruction::<WORD_SIZE, 2>(x);
            jolt_instruction_test!(instruction);
        }
        // ALIGN = 4
        for _ in 0..256 {
            let x = rng.next_u64();
            let instruction = AssertAlignedMemoryAccessInstruction::<WORD_SIZE, 4>(x);
            jolt_instruction_test!(instruction);
        }
    }
}
