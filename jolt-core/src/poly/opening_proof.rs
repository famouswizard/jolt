use rayon::prelude::*;

use crate::{
    field::{JoltField, OptimizedMul},
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{
        thread::unsafe_allocate_zero_vec,
        transcript::{AppendToTranscript, ProofTranscript},
    },
};

use super::{
    commitment::commitment_scheme::CommitmentScheme,
    dense_mlpoly::DensePolynomial,
    unipoly::{CompressedUniPoly, UniPoly},
};

pub struct ProverOpening<F: JoltField> {
    pub polynomial: DensePolynomial<F>,
    pub eq_poly: DensePolynomial<F>,
    pub opening_point: Vec<F>,
    pub claim: F,
    pub num_sumcheck_rounds: usize,
}

pub struct VerifierOpening<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub commitment: PCS::Commitment,
    pub opening_point: Vec<F>,
    pub claim: F,
    pub num_sumcheck_rounds: usize,
}

impl<F: JoltField> ProverOpening<F> {
    fn new(
        polynomial: DensePolynomial<F>,
        eq_poly: DensePolynomial<F>,
        opening_point: Vec<F>,
        claim: F,
    ) -> Self {
        let num_sumcheck_rounds = polynomial.get_num_vars();
        ProverOpening {
            polynomial,
            eq_poly,
            opening_point,
            claim,
            num_sumcheck_rounds,
        }
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> VerifierOpening<F, PCS> {
    fn new(commitment: PCS::Commitment, opening_point: Vec<F>, claim: F) -> Self {
        let num_sumcheck_rounds = opening_point.len();
        VerifierOpening {
            commitment,
            opening_point,
            claim,
            num_sumcheck_rounds,
        }
    }
}

pub struct ProverOpeningAccumulator<F: JoltField> {
    openings: Vec<ProverOpening<F>>,
}

pub struct VerifierOpeningAccumulator<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    openings: Vec<VerifierOpening<F, PCS>>,
}

impl<F: JoltField> ProverOpeningAccumulator<F> {
    pub fn new() -> Self {
        Self { openings: vec![] }
    }

    pub fn len(&self) -> usize {
        self.openings.len()
    }

    pub fn append(
        &mut self,
        polynomials: &[&DensePolynomial<F>],
        eq_poly: DensePolynomial<F>,
        opening_point: Vec<F>,
        claims: &[&F],
        transcript: &mut ProofTranscript,
    ) {
        assert_eq!(polynomials.len(), claims.len());
        // Generate batching challenge \rho and powers 1,...,\rho^{m-1}
        let rho: F = transcript.challenge_scalar();
        // let rho: F = F::one(); // TODO(moodlezoup)

        let mut rho_powers = vec![F::one()];
        for i in 1..polynomials.len() {
            rho_powers.push(rho_powers[i - 1] * rho);
        }

        let batched_claim = rho_powers
            .iter()
            .zip(claims.iter())
            .map(|(scalar, eval)| *scalar * *eval)
            .sum();

        let num_chunks = rayon::current_num_threads().next_power_of_two();
        let chunk_size = (1 << opening_point.len()) / num_chunks;
        let f_batched = (0..num_chunks)
            .into_par_iter()
            .flat_map_iter(|chunk_index| {
                let mut chunk = unsafe_allocate_zero_vec::<F>(chunk_size);
                for (coeff, poly) in rho_powers.iter().zip(polynomials.iter()) {
                    for (rlc, poly_eval) in chunk
                        .iter_mut()
                        .zip(poly.evals_ref()[chunk_index * chunk_size..].iter())
                    {
                        *rlc += poly_eval.mul_01_optimized(*coeff);
                    }
                }
                chunk
            })
            .collect::<Vec<_>>();

        let batched_poly = DensePolynomial::new(f_batched);

        self.openings.push(ProverOpening::new(
            batched_poly,
            eq_poly,
            opening_point,
            batched_claim,
        ));
    }

    pub fn par_extend<I: IntoParallelIterator<Item = ProverOpening<F>>>(&mut self, iter: I) {
        self.openings.par_extend(iter);
    }

    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::reduce_and_prove")]
    pub fn reduce_and_prove<PCS: CommitmentScheme<Field = F>>(
        &mut self,
        pcs_setup: &PCS::Setup,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F>, Vec<F>, PCS::Proof) {
        // Generate coefficients for random linear combination
        let rho: F = transcript.challenge_scalar();
        let mut rho_powers = vec![F::one()];
        for i in 1..self.openings.len() {
            rho_powers.push(rho_powers[i - 1] * rho);
        }

        let (sumcheck_proof, r_sumcheck, sumcheck_claims) =
            self.prove_batch_opening_reduction(&rho_powers, transcript);

        transcript.append_scalars(&sumcheck_claims);
        let gamma: F = transcript.challenge_scalar();
        let mut gamma_powers = vec![F::one()];
        for i in 1..self.openings.len() {
            gamma_powers.push(gamma_powers[i - 1] * gamma);
        }

        let max_len = self
            .openings
            .iter()
            .map(|opening| opening.polynomial.len())
            .max()
            .unwrap();
        let num_chunks = rayon::current_num_threads().next_power_of_two();
        let chunk_size = max_len / num_chunks;
        assert!(chunk_size > 0);

        let joint_poly: Vec<F> = (0..num_chunks)
            .into_par_iter()
            .flat_map_iter(|chunk_index| {
                let mut chunk = unsafe_allocate_zero_vec(chunk_size);
                for (coeff, opening) in gamma_powers.iter().zip(self.openings.iter()) {
                    if chunk_index * chunk_size >= opening.polynomial.len() {
                        continue;
                    }
                    for (rlc, poly_eval) in chunk
                        .iter_mut()
                        .zip(opening.polynomial.Z[chunk_index * chunk_size..].iter())
                    {
                        *rlc += coeff.mul_01_optimized(*poly_eval);
                    }
                }
                chunk
            })
            .collect();
        let joint_poly = DensePolynomial::new(joint_poly);

        // // todo!("scale opening claims");
        // // Compute joint claim = ∑ᵢ γⁱ⋅ claimᵢ
        // let joint_claim: F = gamma_powers
        //     .iter()
        //     .zip(self.openings.iter())
        //     .map(|(scalar, opening)| *scalar * opening.claim)
        //     .sum();

        let joint_opening_proof = PCS::prove(pcs_setup, &joint_poly, &r_sumcheck, transcript);

        (sumcheck_proof, sumcheck_claims, joint_opening_proof)
    }

    #[tracing::instrument(skip_all, name = "prove_batch_opening_reduction")]
    pub fn prove_batch_opening_reduction(
        &mut self,
        coeffs: &[F],
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F>, Vec<F>, Vec<F>) {
        let max_num_vars = self
            .openings
            .iter()
            .map(|opening| opening.polynomial.get_num_vars())
            .max()
            .unwrap();

        let mut e: F = coeffs
            .par_iter()
            .zip(self.openings.par_iter())
            .map(|(coeff, opening)| {
                let scaled_claim = if opening.polynomial.get_num_vars() != max_num_vars {
                    F::from_u64(1 << (max_num_vars - opening.polynomial.get_num_vars())).unwrap()
                        * opening.claim
                } else {
                    opening.claim
                };
                scaled_claim * coeff
            })
            .sum();

        let mut r: Vec<F> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();
        let mut bound_polys: Vec<Option<DensePolynomial<F>>> = vec![None; self.openings.len()];

        for round in 0..max_num_vars {
            let remaining_rounds = max_num_vars - round;
            let uni_poly = self.compute_quadratic(coeffs, remaining_rounds, &mut bound_polys, e);
            let compressed_poly = uni_poly.compress();

            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            r.push(r_j);

            self.bind(remaining_rounds, &mut bound_polys, r_j);

            e = uni_poly.evaluate(&r_j);
            compressed_polys.push(compressed_poly);
        }

        let claims: Vec<_> = bound_polys
            .into_iter()
            .map(|poly| {
                let poly = poly.unwrap();
                debug_assert_eq!(poly.len(), 1);
                poly[0]
            })
            .collect();

        (SumcheckInstanceProof::new(compressed_polys), r, claims)
    }

    #[tracing::instrument(skip_all, name = "compute_quadratic")]
    fn compute_quadratic(
        &self,
        coeffs: &[F],
        remaining_sumcheck_rounds: usize,
        bound_polys: &mut Vec<Option<DensePolynomial<F>>>,
        previous_round_claim: F,
    ) -> UniPoly<F> {
        let evals: Vec<(F, F)> = self
            .openings
            .par_iter()
            .zip(bound_polys.par_iter())
            .map(|(opening, bound_poly)| {
                if remaining_sumcheck_rounds <= opening.num_sumcheck_rounds {
                    let poly = bound_poly.as_ref().unwrap_or(&opening.polynomial);
                    let mle_half = poly.len() / 2;
                    let eval_0: F = (0..mle_half)
                        .into_iter()
                        .map(|i| poly[i].mul_01_optimized(opening.eq_poly[i]))
                        .sum();
                    let eval_2: F = (0..mle_half)
                        .into_iter()
                        .map(|i| {
                            let poly_bound_point =
                                poly[i + mle_half] + poly[i + mle_half] - poly[i];
                            let eq_bound_point = opening.eq_poly[i + mle_half]
                                + opening.eq_poly[i + mle_half]
                                - opening.eq_poly[i];
                            poly_bound_point.mul_01_optimized(eq_bound_point)
                        })
                        .sum();
                    (eval_0, eval_2)
                } else {
                    debug_assert!(bound_poly.is_none());
                    let remaining_variables =
                        remaining_sumcheck_rounds - opening.num_sumcheck_rounds - 1;
                    let scaled_claim =
                        F::from_u64(1 << remaining_variables).unwrap() * opening.claim;
                    (scaled_claim, scaled_claim)
                }
            })
            .collect();

        let evals_combined_0: F = (0..evals.len()).map(|i| evals[i].0 * coeffs[i]).sum();
        let evals_combined_2: F = (0..evals.len()).map(|i| evals[i].1 * coeffs[i]).sum();
        let evals = vec![
            evals_combined_0,
            previous_round_claim - evals_combined_0,
            evals_combined_2,
        ];

        UniPoly::from_evals(&evals)
    }

    #[tracing::instrument(skip_all, name = "bind")]
    fn bind(
        &mut self,
        remaining_sumcheck_rounds: usize,
        bound_polys: &mut Vec<Option<DensePolynomial<F>>>,
        r_j: F,
    ) {
        self.openings
            .par_iter_mut()
            .zip(bound_polys.par_iter_mut())
            .for_each(|(opening, bound_poly)| {
                if remaining_sumcheck_rounds <= opening.num_sumcheck_rounds {
                    match bound_poly {
                        Some(bound_poly) => {
                            rayon::join(
                                || opening.eq_poly.bound_poly_var_top(&r_j),
                                || bound_poly.bound_poly_var_top(&r_j),
                            );
                        }
                        None => {
                            *bound_poly = rayon::join(
                                || opening.eq_poly.bound_poly_var_top(&r_j),
                                || Some(opening.polynomial.new_poly_from_bound_poly_var_top(&r_j)),
                            )
                            .1;
                        }
                    };
                }
            });
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> VerifierOpeningAccumulator<F, PCS> {
    pub fn new() -> Self {
        Self { openings: vec![] }
    }

    pub fn len(&self) -> usize {
        self.openings.len()
    }

    pub fn append(
        &mut self,
        commitments: &[&PCS::Commitment],
        opening_point: Vec<F>,
        claims: &[&F],
        transcript: &mut ProofTranscript,
    ) {
        assert_eq!(commitments.len(), claims.len());
        let _: F = transcript.challenge_scalar();
        // todo!("Compute RLC commitment/claim");
        // self.openings
        //     .push(VerifierOpening::new(commitment, opening_point, claim));
    }

    pub fn par_extend<I: IntoParallelIterator<Item = VerifierOpening<F, PCS>>>(&mut self, iter: I) {
        self.openings.par_extend(iter);
    }

    pub fn reduce_and_verify(
        &self,
        pcs_setup: &PCS::Setup,
        reduction_sumcheck_proof: SumcheckInstanceProof<F>,
        reduced_opening_proof: PCS::Proof,
        transcript: &mut ProofTranscript,
    ) {
        todo!();
    }
}
