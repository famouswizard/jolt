use std::marker::PhantomData;

use ark_ff::PrimeField;
use ark_std::log2;

use enum_dispatch::enum_dispatch;
use itertools::Chunk;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use crate::poly::dense_mlpoly::DensePolynomial;
use crate::utils::math::Math;
use crate::utils::split_bits;

// use super::jolt_strategy::{InstructionStrategy, JoltStrategy, SubtableStrategy};

// ==================== INSTRUCTIONS ====================

trait JoltInstruction<F: PrimeField> {
  // TODO: C, M
  //   type Subtables;

  fn combine_lookups(vals: &[F]) -> F;
  fn g_poly_degree() -> usize;
}

#[enum_dispatch]
pub trait ChunkIndices {
  fn to_indices<const C: usize, const LOG_M: usize>(&self) -> [usize; C];
}

pub trait Opcode {
  fn to_opcode(&self) -> u8;
}

#[repr(u8)]
#[derive(Copy, Clone)]
#[enum_dispatch(ChunkIndices)]
pub enum TestInstructionSet {
  XOR(XORInstruction),
  EQ(EQInstruction),
  LT(LTInstruction),
  NOT(NOTInstruction),
}

impl Opcode for TestInstructionSet {
  fn to_opcode(&self) -> u8 {
    unsafe { *<*const _>::from(self).cast::<u8>() }
  }
}

#[derive(Copy, Clone)]
pub struct XORInstruction(u64, u64);
#[derive(Copy, Clone)]
pub struct EQInstruction(u64, u64);
#[derive(Copy, Clone)]
pub struct LTInstruction(u64, u64);
#[derive(Copy, Clone)]
pub struct NOTInstruction(u64);

impl<F: PrimeField> JoltInstruction<F> for XORInstruction {
  // TODO: assocsiated tuple vs associated enum vs method
  //   type Subtables = (XORSubtable<F>);

  fn combine_lookups(vals: &[F]) -> F {
    unimplemented!("TODO");
  }

  fn g_poly_degree() -> usize {
    1
  }
}

impl ChunkIndices for XORInstruction {
  fn to_indices<const C: usize, const LOG_M: usize>(&self) -> [usize; C] {
    unimplemented!("TODO");
  }
}

impl<F: PrimeField> JoltInstruction<F> for EQInstruction {
  fn combine_lookups(vals: &[F]) -> F {
    unimplemented!("TODO");
  }

  fn g_poly_degree() -> usize {
    unimplemented!("TODO");
  }
}

impl ChunkIndices for EQInstruction {
  fn to_indices<const C: usize, const LOG_M: usize>(&self) -> [usize; C] {
    unimplemented!("TODO");
  }
}

impl<F: PrimeField> JoltInstruction<F> for LTInstruction {
  fn combine_lookups(vals: &[F]) -> F {
    unimplemented!("TODO");
  }

  fn g_poly_degree() -> usize {
    unimplemented!("TODO");
  }
}

impl ChunkIndices for LTInstruction {
  fn to_indices<const C: usize, const LOG_M: usize>(&self) -> [usize; C] {
    unimplemented!("TODO");
  }
}

impl<F: PrimeField> JoltInstruction<F> for NOTInstruction {
  fn combine_lookups(vals: &[F]) -> F {
    unimplemented!("TODO");
  }

  fn g_poly_degree() -> usize {
    1
  }
}

impl ChunkIndices for NOTInstruction {
  fn to_indices<const C: usize, const LOG_M: usize>(&self) -> [usize; C] {
    unimplemented!("TODO");
  }
}

// ==================== SUBTABLES ====================

#[enum_dispatch]
pub trait LassoSubtable<F: PrimeField> {
  // TODO: M

  fn materialize(&self) -> Vec<F>;
  fn evaluate_mle(&self, point: &[F]) -> F;
}

#[enum_dispatch(LassoSubtable<F>)]
#[derive(EnumCountMacro, EnumIter)]
pub enum TestSubtables<F: PrimeField> {
  XOR(XORSubtable<F>),
  EQ(EQSubtable<F>),
}

#[derive(Default)]
pub struct XORSubtable<F: PrimeField> {
  _field: PhantomData<F>,
}
#[derive(Default)]
pub struct EQSubtable<F: PrimeField> {
  _field: PhantomData<F>,
}

impl<F: PrimeField> LassoSubtable<F> for XORSubtable<F> {
  fn materialize(&self) -> Vec<F> {
    unimplemented!("TODO");
  }

  fn evaluate_mle(&self, point: &[F]) -> F {
    unimplemented!("TODO");
  }
}

impl<F: PrimeField> LassoSubtable<F> for EQSubtable<F> {
  fn materialize(&self) -> Vec<F> {
    unimplemented!("TODO");
  }

  fn evaluate_mle(&self, point: &[F]) -> F {
    unimplemented!("TODO");
  }
}

// ==================== JOLT ====================
pub trait Jolt<F: PrimeField, const C: usize, const LOG_M: usize> {
  type InstructionSet: ChunkIndices + Opcode;
  type Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount;

  const NUM_SUBTABLES: usize = Self::Subtables::COUNT;
  const NUM_MEMORIES: usize = C * Self::Subtables::COUNT;

  fn prove(ops: Vec<Self::InstructionSet>) {
    Self::polynomialize(ops);
    // unimplemented!("TODO");
  }

  fn polynomialize(ops: Vec<Self::InstructionSet>) {
    // let chunked_indices: Vec<[usize; C]> =
    //   ops.iter().map(|op| op.to_indices::<C, LOG_M>()).collect();
    let ops_u8: Vec<u8> = ops.iter().map(|op| op.to_opcode()).collect();
    println!("{:?}", ops_u8);

    // let mut dim_usize: Vec<Vec<usize>> = Vec::with_capacity(C);

    // let mut dim: Vec<DensePolynomial<F>> = Vec::with_capacity(C);
    // let mut read: Vec<DensePolynomial<F>> = Vec::with_capacity(C);
    // let mut r#final: Vec<DensePolynomial<F>> = Vec::with_capacity(C);

    // let s = ops.len().next_power_of_two();

    // for i in 0..C {
    //   let mut final_timestamps = vec![0usize; LOG_M.pow2()];
    //   let mut read_timestamps = vec![0usize; s];
    // }
  }

  fn materialize_subtables() -> Vec<Vec<F>> {
    let mut subtables: Vec<Vec<F>> = Vec::with_capacity(Self::Subtables::COUNT);
    for subtable in Self::Subtables::iter() {
      subtables.push(subtable.materialize());
    }
    subtables
  }
}

pub struct TestJoltVM<F: PrimeField, const C: usize, const LOG_M: usize> {
  _field: PhantomData<F>,
}

impl<F: PrimeField, const C: usize, const LOG_M: usize> Jolt<F, C, LOG_M>
  for TestJoltVM<F, C, LOG_M>
{
  type InstructionSet = TestInstructionSet;
  type Subtables = TestSubtables<F>;
}

// ==================== TEST ====================

#[cfg(test)]
mod tests {
  use ark_curve25519::{EdwardsProjective, Fr};
  use ark_ff::PrimeField;
  use ark_std::{log2, test_rng, One, Zero};
  use merlin::Transcript;
  use rand_chacha::rand_core::RngCore;

  use crate::{
    jolt::test_vm::{EQInstruction, Jolt, TestInstructionSet, TestJoltVM, XORInstruction},
    utils::{index_to_field_bitvector, random::RandomTape, split_bits},
  };

  //   pub fn gen_indices<const C: usize>(sparsity: usize, memory_size: usize) -> Vec<Vec<usize>> {
  //     let mut rng = test_rng();
  //     let mut all_indices: Vec<Vec<usize>> = Vec::new();
  //     for _ in 0..sparsity {
  //       let indices = vec![rng.next_u64() as usize % memory_size; C];
  //       all_indices.push(indices);
  //     }
  //     all_indices
  //   }

  //   pub fn gen_random_point<F: PrimeField>(memory_bits: usize) -> Vec<F> {
  //     let mut rng = test_rng();
  //     let mut r_i: Vec<F> = Vec::with_capacity(memory_bits);
  //     for _ in 0..memory_bits {
  //       r_i.push(F::rand(&mut rng));
  //     }
  //     r_i
  //   }

  #[test]
  fn e2e() {
    const C: usize = 4;
    // const S: usize = 1 << 8;
    const M: usize = 1 << 16;

    // let log_m = log2(M) as usize;
    // let log_s: usize = log2(S) as usize;

    TestJoltVM::<Fr, C, 16>::prove(vec![
      TestInstructionSet::XOR(XORInstruction(420, 69)),
      TestInstructionSet::EQ(EQInstruction(420, 69)),
      TestInstructionSet::EQ(EQInstruction(420, 420)),
    ]);
  }
}
