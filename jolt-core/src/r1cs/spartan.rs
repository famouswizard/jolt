#![allow(clippy::len_without_is_empty)]

// use core::range;
use std::marker::PhantomData;

use crate::field::JoltField;
use crate::jolt::vm::JoltCommitments;
use crate::jolt::vm::JoltPolynomials;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::r1cs::key::UniformSpartanKey;
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;

use crate::utils::transcript::Transcript;
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;

use rayon::prelude::*;
use thiserror::Error;

use crate::{
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial},
    subprotocols::sumcheck::SumcheckInstanceProof,
    r1cs::special_polys::eq_plus_one,
};

use super::builder::CombinedUniformBuilder;
use super::inputs::ConstraintInput;

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum SpartanError {
    /// returned if the supplied row or col in (row,col,val) tuple is out of range
    #[error("InvalidIndex")]
    InvalidIndex,

    /// returned when an invalid sum-check proof is provided
    #[error("InvalidSumcheckProof")]
    InvalidSumcheckProof,

    /// returned when the recusive sumcheck proof fails
    #[error("InvalidOuterSumcheckProof")]
    InvalidOuterSumcheckProof,

    /// returned when the final sumcheck opening proof fails
    #[error("InvalidOuterSumcheckClaim")]
    InvalidOuterSumcheckClaim,

    /// returned when the recusive sumcheck proof fails
    #[error("InvalidInnerSumcheckProof")]
    InvalidInnerSumcheckProof,

    /// returned when the final sumcheck opening proof fails
    #[error("InvalidInnerSumcheckClaim")]
    InvalidInnerSumcheckClaim,

    /// returned if the supplied witness is not of the right length
    #[error("InvalidWitnessLength")]
    InvalidWitnessLength,

    /// returned when an invalid PCS proof is provided
    #[error("InvalidPCSProof")]
    InvalidPCSProof,
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct UniformSpartanProof<
    const C: usize,
    I: ConstraintInput,
    F: JoltField,
    ProofTranscript: Transcript,
> {
    _inputs: PhantomData<I>,
    pub(crate) outer_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) outer_sumcheck_claims: (F, F, F),
    pub(crate) inner_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) claimed_witness_evals: Vec<F>,
    _marker: PhantomData<ProofTranscript>,
}

impl<const C: usize, I, F, ProofTranscript> UniformSpartanProof<C, I, F, ProofTranscript>
where
    I: ConstraintInput,
    F: JoltField,
    ProofTranscript: Transcript,
{
    #[tracing::instrument(skip_all, name = "Spartan::setup")]
    pub fn setup(
        constraint_builder: &CombinedUniformBuilder<C, F, I>,
        padded_num_steps: usize,
    ) -> UniformSpartanKey<C, I, F> {
        assert_eq!(
            padded_num_steps,
            constraint_builder.uniform_repeat().next_power_of_two()
        );
        UniformSpartanKey::from_builder(constraint_builder)
    }

    #[tracing::instrument(skip_all, name = "Spartan::prove")]
    pub fn prove<PCS>(
        constraint_builder: &CombinedUniformBuilder<C, F, I>,
        key: &UniformSpartanKey<C, I, F>,
        polynomials: &JoltPolynomials<F>,
        opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<Self, SpartanError>
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
    {
        let flattened_polys: Vec<&DensePolynomial<F>> = I::flatten::<C>()
            .iter()
            .map(|var| var.get_ref(polynomials))
            .collect();

        let num_rounds_x = key.num_rows_total().log_2();
        let num_rounds_y = key.num_cols_total().log_2();

        // outer sum-check
        let tau = (0..num_rounds_x)
            .map(|_i| transcript.challenge_scalar())
            .collect::<Vec<F>>();
        let mut poly_tau = DensePolynomial::new(EqPolynomial::evals(&tau));

        let (mut az, mut bz, mut cz) =
            constraint_builder.compute_spartan_Az_Bz_Cz::<PCS, ProofTranscript>(&flattened_polys);

        let comb_func_outer = |eq: &F, az: &F, bz: &F, cz: &F| -> F {
            // Below is an optimized form of: eq * (Az * Bz - Cz)
            if az.is_zero() || bz.is_zero() {
                if cz.is_zero() {
                    F::zero()
                } else {
                    *eq * (-(*cz))
                }
            } else {
                let inner = *az * *bz - *cz;
                if inner.is_zero() {
                    F::zero()
                } else {
                    *eq * inner
                }
            }
        };

        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) =
            SumcheckInstanceProof::prove_spartan_cubic::<_>(
                &F::zero(), // claim is zero
                num_rounds_x,
                &mut poly_tau,
                &mut az,
                &mut bz,
                &mut cz,
                comb_func_outer,
                transcript,
            );
        let outer_sumcheck_r: Vec<F> = outer_sumcheck_r.into_iter().rev().collect();
        drop_in_background_thread((az, bz, cz, poly_tau));

        // claims from the end of sum-check
        // claim_Az is the (scalar) value v_A = \sum_y A(r_x, y) * z(r_x) where r_x is the sumcheck randomness
        let (claim_Az, claim_Bz, claim_Cz): (F, F, F) = (
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
            outer_sumcheck_claims[3],
        );

        ProofTranscript::append_scalars(transcript, [claim_Az, claim_Bz, claim_Cz].as_slice());

        // inner sum-check
        let r_inner_sumcheck_RLC: F = transcript.challenge_scalar();
        let claim_inner_joint = claim_Az
            + r_inner_sumcheck_RLC * claim_Bz
            + r_inner_sumcheck_RLC * r_inner_sumcheck_RLC * claim_Cz;

        // Crush:
        let num_steps_padded = constraint_builder.uniform_repeat().next_power_of_two();
        let num_steps_bits_ = constraint_builder
            .uniform_repeat()
            .next_power_of_two()
            .ilog2() as usize;
        let num_constraints_bits = key.num_cons_total.log_2() - num_steps_bits_;
        let r_x_step = &outer_sumcheck_r[num_constraints_bits..];

        let mut z: Vec<F> = flattened_polys.clone().into_iter().map(|poly| {
            let mut resized = poly.Z.clone();
            resized.resize(poly.len().next_power_of_two(), F::zero());
            resized
        }).flatten().collect();
        z.resize(z.len().next_power_of_two(), F::zero());

        let is_last_step = EqPolynomial::new(r_x_step.to_vec()).evaluate(&vec![F::one(); r_x_step.len()]);
        let eq_rx_step = EqPolynomial::evals(r_x_step);

        // // Evals with binding. Doesn't ignore when r_x_step is the last step. 
        // for r_s in r_x_step.iter().rev() {
        //     poly_z.bound_poly_var_bot(r_s);
        // }
        // let mut evals = poly_z.evals();
        // evals.push(F::one() - is_last_step); // ARASU: IGNORE LAST STEP? 
        // evals.resize(evals.len().next_power_of_two() * 1, F::zero());

        // Evals straightfoward
        let mut evals: Vec<F> = (0..key.num_vars_uniform()) // until the constant (which is not included)
        .map(|y_var| {
            (0..(num_steps_padded-1)) // Ignore the last step
                .map(|t| z[y_var * num_steps_padded + t] * eq_rx_step[t])
                .sum()
        })
        .collect();
        evals.resize(evals.len().next_power_of_two(), F::zero());
        evals.push(F::one() - is_last_step); // Constant, ignores the last step.  
        evals.resize(evals.len().next_power_of_two(), F::zero());

        let n_bits_ts = r_x_step.len();
        let eq_plus_one_rx_step: Vec<F> = (0..num_steps_padded)
            .map(|t| eq_plus_one(r_x_step, &crate::utils::index_to_field_bitvector(t, n_bits_ts), n_bits_ts))
            .collect();

        let mut evals_shifted = (0..key.num_vars_uniform())
            .map(|y_var: usize| {
                (0..num_steps_padded-1)
                    .map(|t| 
                        z[y_var * num_steps_padded + t] * eq_plus_one_rx_step[t] 
                    )
                    .sum::<F>()
            })
            .collect::<Vec<F>>();
        evals_shifted.resize(evals.len(), F::zero());

        let poly_z = DensePolynomial::new(evals.into_iter().chain(evals_shifted.into_iter()).collect());

        // this is the polynomial extended from the vector r_A * A(r_x, y) + r_B * B(r_x, y) + r_C * C(r_x, y) for all y
        let num_steps_bits = constraint_builder
            .uniform_repeat()
            .next_power_of_two()
            .ilog2();
        let (rx_con, rx_ts) =
            outer_sumcheck_r.split_at(outer_sumcheck_r.len() - num_steps_bits as usize);
        let poly_ABC =
            DensePolynomial::new(key.evaluate_r1cs_mle_rlc(rx_con, rx_ts, r_inner_sumcheck_RLC));
        assert_eq!(poly_z.len(), poly_ABC.len());
        assert_eq!(poly_ABC.len(), key.num_vars_uniform().next_power_of_two() * 4); // *4 to support cross_step constraints

        let num_rounds = poly_ABC.len().log_2();
        let mut polys = vec![poly_ABC, poly_z]; 
        let comb_func = |poly_evals: &[F]| -> F {
            assert_eq!(poly_evals.len(), 2);
            poly_evals[0] * poly_evals[1]
        };
        let (inner_sumcheck_proof, inner_sumcheck_r, _claims_inner) =
            SumcheckInstanceProof::prove_arbitrary(
                &claim_inner_joint, 
                num_rounds, 
                &mut polys, 
                comb_func, 
                2, 
                transcript);
        
        drop_in_background_thread(polys);

        // Crush:
        let r_z = r_x_step; 

        let chi = EqPolynomial::evals(&r_z);
        let claimed_witness_evals: Vec<_> = flattened_polys
            .par_iter()
            .map(|poly| poly.evaluate_at_chi_low_optimized(&chi))
            .collect();

        opening_accumulator.append(
            &flattened_polys,
            DensePolynomial::new(chi),
            r_z.to_vec(),
            &claimed_witness_evals.iter().collect::<Vec<_>>(),
            transcript,
        );

        // Outer sumcheck claims: [eq(r_x), A(r_x), B(r_x), C(r_x)]
        let outer_sumcheck_claims = (
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
            outer_sumcheck_claims[3],
        );
        Ok(UniformSpartanProof {
            _inputs: PhantomData,
            outer_sumcheck_proof,
            outer_sumcheck_claims,
            inner_sumcheck_proof,
            claimed_witness_evals,
            _marker: PhantomData,
        })
    }

    #[tracing::instrument(skip_all, name = "Spartan::verify")]
    pub fn verify<PCS>(
        &self,
        key: &UniformSpartanKey<C, I, F>,
        commitments: &JoltCommitments<PCS, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), SpartanError>
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        let num_rounds_x = key.num_rows_total().log_2();
        let num_rounds_y = key.num_cols_total().log_2();

        // outer sum-check
        let tau = (0..num_rounds_x)
            .map(|_i| transcript.challenge_scalar())
            .collect::<Vec<F>>();

        let (claim_outer_final, r_x) = self
            .outer_sumcheck_proof
            .verify(F::zero(), num_rounds_x, 3, transcript)
            .map_err(|_| SpartanError::InvalidOuterSumcheckProof)?;

        // Outer sumcheck is bound from the top, reverse the fiat shamir randomness
        let r_x: Vec<F> = r_x.into_iter().rev().collect();

        // verify claim_outer_final
        let (claim_Az, claim_Bz, claim_Cz) = self.outer_sumcheck_claims;
        let taus_bound_rx = EqPolynomial::new(tau).evaluate(&r_x);
        let claim_outer_final_expected = taus_bound_rx * (claim_Az * claim_Bz - claim_Cz);
        if claim_outer_final != claim_outer_final_expected {
            return Err(SpartanError::InvalidOuterSumcheckClaim);
        }

        transcript.append_scalars(
            [
                self.outer_sumcheck_claims.0,
                self.outer_sumcheck_claims.1,
                self.outer_sumcheck_claims.2,
            ]
            .as_slice(),
        );

        // inner sum-check
        let r_inner_sumcheck_RLC: F = transcript.challenge_scalar();
        let claim_inner_joint = self.outer_sumcheck_claims.0
            + r_inner_sumcheck_RLC * self.outer_sumcheck_claims.1
            + r_inner_sumcheck_RLC * r_inner_sumcheck_RLC * self.outer_sumcheck_claims.2;


        let num_rounds = (key.num_vars_uniform() * 2).next_power_of_two().log_2();
        let (claim_inner_final, inner_sumcheck_r) = self
            .inner_sumcheck_proof
            .verify(claim_inner_joint, num_rounds, 2, transcript) 
            .map_err(|_| SpartanError::InvalidInnerSumcheckProof)?;

        // n_prefix = n_segments + 1
        let n_prefix = key.uniform_r1cs.num_vars.next_power_of_two().log_2() + 1;

        // Crush: 
        let n_constraint_bits_uniform = key.uniform_r1cs.num_rows.next_power_of_two().log_2();
        let outer_sumcheck_r_step = &r_x[n_constraint_bits_uniform..];
        let y_prime = [inner_sumcheck_r.to_owned(), outer_sumcheck_r_step.to_owned()].concat();
        let eval_Z = key.evaluate_z_mle(&self.claimed_witness_evals, &y_prime);

        // Crush: 
        let r = [r_x.clone(), y_prime].concat();
        let (eval_a, eval_b, eval_c) = key.evaluate_r1cs_matrix_mles(&r);

        let left_expected = eval_a
            + r_inner_sumcheck_RLC * eval_b
            + r_inner_sumcheck_RLC * r_inner_sumcheck_RLC * eval_c;
        let right_expected = eval_Z;
        let claim_inner_final_expected = left_expected * right_expected;

        assert_eq!(claim_inner_final, claim_inner_final_expected);
        println!("claim_inner_final: {:?}", claim_inner_final);
        println!("claim_inner_final_expected: {:?}", claim_inner_final_expected);
        assert!(false); 
        if claim_inner_final != claim_inner_final_expected {
            return Err(SpartanError::InvalidInnerSumcheckClaim);
        }

        let flattened_commitments: Vec<_> = I::flatten::<C>()
            .iter()
            .map(|var| var.get_ref(commitments))
            .collect();
        // Crush: 
        let r_y_point = &r_x[n_constraint_bits_uniform..];
        opening_accumulator.append(
            &flattened_commitments,
            r_y_point.to_vec(),
            &self.claimed_witness_evals.iter().collect::<Vec<_>>(),
            transcript,
        );

        Ok(())
    }
}

// #[cfg(test)]
// mod test {
//     use ark_bn254::Fr;
//     use ark_std::One;

//     use crate::poly::commitment::{commitment_scheme::CommitShape, hyrax::HyraxScheme};

//     use super::*;

//     #[test]
//     fn integration() {
//         let (builder, key) = simp_test_builder_key();
//         let witness_segments: Vec<Vec<Fr>> = vec![
//             vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)], /* Q */
//             vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)], /* R */
//             vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)], /* S */
//         ];

//         // Create a witness and commit
//         let witness_segments_ref: Vec<&[Fr]> = witness_segments
//             .iter()
//             .map(|segment| segment.as_slice())
//             .collect();
//         let gens = HyraxScheme::setup(&[CommitShape::new(16, BatchType::Small)]);
//         let witness_commitment =
//             HyraxScheme::batch_commit(&witness_segments_ref, &gens, BatchType::Small);

//         // Prove spartan!
//         let mut prover_transcript = ProofTranscript::new(b"stuff");
//         let proof =
//             UniformSpartanProof::<Fr, HyraxScheme<ark_bn254::G1Projective>>::prove_precommitted::<
//                 SimpTestIn,
//             >(
//                 &gens,
//                 builder,
//                 &key,
//                 witness_segments,
//                 todo!("opening accumulator"),
//                 &mut prover_transcript,
//             )
//             .unwrap();

//         let mut verifier_transcript = ProofTranscript::new(b"stuff");
//         let witness_commitment_ref: Vec<&_> = witness_commitment.iter().collect();
//         proof
//             .verify_precommitted(
//                 &key,
//                 witness_commitment_ref,
//                 &gens,
//                 &mut verifier_transcript,
//             )
//             .expect("Spartan verifier failed");
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand_core::{RngCore, CryptoRng};
    use ark_bn254::Fr as F; // Add this line to import the field type
    use ark_ff::{Zero, One}; // Import the Zero trait

    #[test]
    fn test_shifted_polynomial_evaluations() {
        // Generate a vector z of random field elements of length 128
        let mut rng = rand::thread_rng();
        let z: Vec<F> = (0..128).map(|_| F::from(rng.gen::<u64>())).collect();

        // Resize z to the next power of two
        let mut z_resized = z.clone();
        z_resized.resize(z.len().next_power_of_two() * 2, F::zero());

        println!("z_resized.len(): {:?}", z_resized.len());

        let r_x_step: Vec<F> = vec![F::zero(), F::zero(), F::one(), F::zero()];

        // Create the polynomial from z
        let mut poly_z = DensePolynomial::new(z_resized.clone());
        for r_s in r_x_step.iter().rev() {
            poly_z.bound_poly_var_bot(r_s);
        }
        let evals = poly_z.evals();

        // Create the shifted polynomial from z
        let mut z_shifted: Vec<F> = z[1..].to_vec();
        z_shifted.resize(z.len().next_power_of_two(), F::zero());

        let mut poly_z_shifted = DensePolynomial::new(z_shifted.clone());
        for r_s in r_x_step.iter().rev() {
            poly_z_shifted.bound_poly_var_bot(r_s);
        }
        let evals_shifted = poly_z_shifted.evals();

        // // print the first 10 lines of evals and evals_shifted 
        // for i in 0..4 {
        //     println!("z: {:?}", z); 
        //     println!("evals_shifted: {:?}", evals_shifted); 
        //     // println!("evals[{}]: {:?}, evals_shifted[{}]: {:?}", i, evals[i], i, evals_shifted[i]);
        //     println!("z[{}]: {:?}, evals_shifted[{}]: {:?}", i+1, z[i+1], i, evals_shifted[i]);
        //     // println!("z[{}]: {:?}", i+1, z[i+1]);

        // }

        // print each element of z preceded by index: 
        for i in 0..z.len() {
            println!("z[{}]: {:?}", i, z[i]);
        }
        println!("evals_shifted: {:?}", evals_shifted); 



        // // Evaluate the polynomials at a random point k
        // let k: F = F::random(&mut rng);
        // let eval_at_k = poly_z.evaluate(&k);
        // let eval_shifted_at_k_minus_1 = poly_z_shifted.evaluate(&(k - F::one()));

        // // Check if the evaluations are correct
        // assert_eq!(eval_at_k, eval_shifted_at_k_minus_1);
    }
    #[test]
    fn test_eq_polynomial_evals() {
        // Generate a random vector of length 8
        let mut rng = rand::thread_rng();
        let random_vector: Vec<F> = (0..8).map(|_| F::from(rng.gen::<u64>())).collect();

        // generate all 1s vector of lenght 8 
        let all_ones_vector: Vec<F> = (0..8).map(|_| F::one()).collect();

        // Run EqPolynomial::evals on the random vector
        let eq_evals = EqPolynomial::evals(&random_vector);
        let all_ones_evals = EqPolynomial::evals(&all_ones_vector);

        // // Print the random vector and its evaluations
        // for i in 0..random_vector.len() {
        //     println!("random_vector[{}]: {:?}", i, random_vector[i]);
        // }
        // println!("eq_evals: {:?}", eq_evals);
        println!("all_ones_evals.last(): {:?}", all_ones_evals[2]);

        // // Check if the evaluations are correct (this is a placeholder, you should replace it with actual checks)
        // assert_eq!(eq_evals.len(), random_vector.len().next_power_of_two());
    }
}