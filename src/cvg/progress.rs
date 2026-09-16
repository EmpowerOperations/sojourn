//! What the search has in hand, and what it cost to get.
//!
//! The opening's state is a *value*, not a field. Every step of the opening
//! takes a [`Progress`], adds what it found, and hands it back; nothing
//! changes behind a caller's back and there is nothing to lock. What the
//! opening does next — whether to escalate — is read off the value, and what
//! it ends with is what the region holds.

use std::collections::VecDeque;

use crate::Point;

/// How many of the points in hand are kept.
///
/// What reads them: the region, which holds them, and a design's walker,
/// which starts its chains from a pick across them. So the history has one
/// job, to be a fair sample of where the region has been seen, and a thousand
/// recent points do that as well as a million. Unbounded, a brute-force
/// batch at two hundred variables could leave a region holding far more than
/// anything reads.
pub(crate) const RECENT_POINTS: usize = 1_024;

/// One round of uniform proposals: what landed and what it cost.
///
/// Returned by every rung that proposes candidates at random — the probe, a
/// design's round, brute force — and absorbed into a [`Progress`]. The
/// walker does not produce one: its output is not a trial of the region.
#[derive(Debug, Default)]
pub(crate) struct Trial {
    /// The feasible points kept from this round.
    pub(crate) points: Vec<Point>,
    /// Candidates judged to get them.
    pub(crate) proposed: u64,
}

/// Everything the search has in hand.
///
/// A value: every method takes `self` and gives it back, so the opening
/// reads `progress = progress.absorb(trial)` and there is no other way to
/// change it.
#[derive(Debug, Default)]
pub(crate) struct Progress {
    /// The most recent [`RECENT_POINTS`] feasible points in hand, from any
    /// source, oldest at the front. What the region holds and a design's
    /// walker starts from. A window: once full, a point leaves the front for
    /// every one that arrives at the back.
    points: VecDeque<Point>,
    /// Uniform candidates judged so far, and how many points were kept from
    /// them. The walker's output and the solver's witness count as points,
    /// never as trials, so this stays a statement about the region.
    proposed: u64,
    landed: usize,
}

impl Progress {
    pub(crate) fn empty() -> Self {
        Self::default()
    }

    /// Takes in a round of uniform proposals: its points, and its cost.
    #[must_use]
    pub(crate) fn absorb(mut self, trial: Trial) -> Self {
        self.proposed += trial.proposed;
        self.landed += trial.points.len();
        self.remember(trial.points);
        self
    }

    /// Takes in points that were not proposed at random — walked to, or
    /// handed over by a solver — so they count as points and not as trials.
    #[must_use]
    pub(crate) fn extend(mut self, points: Vec<Point>) -> Self {
        self.remember(points);
        self
    }

    /// Appends at the back and lets the oldest go from the front, so the
    /// window never exceeds [`RECENT_POINTS`].
    fn remember(&mut self, points: Vec<Point>) {
        for point in points {
            if self.points.len() == RECENT_POINTS {
                self.points.pop_front();
            }
            self.points.push_back(point);
        }
    }

    /// The window of recent points, oldest first.
    pub(crate) const fn points(&self) -> &VecDeque<Point> {
        &self.points
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub(crate) const fn proposed(&self) -> u64 {
        self.proposed
    }

    pub(crate) const fn landed(&self) -> usize {
        self.landed
    }

    /// The window of recent points, oldest first, as what the region holds.
    pub(crate) fn into_points(self) -> Vec<Point> {
        self.points.into()
    }
}

#[cfg(test)]
mod tests {
    use super::{Progress, Trial};

    fn trial(points: usize, proposed: u64) -> Trial {
        Trial {
            points: (0..points)
                .map(|i| vec![f64::from(u8::try_from(i).unwrap())])
                .collect(),
            proposed,
        }
    }

    #[test]
    fn absorbing_counts_trials_and_extending_does_not() {
        let progress = Progress::empty()
            .absorb(trial(2, 1_000))
            .extend(vec![vec![9.0], vec![8.0], vec![7.0]])
            .absorb(trial(1, 500));

        assert_eq!(progress.points().len(), 6);
        assert_eq!(progress.proposed(), 1_500);
        assert_eq!(progress.landed(), 3, "walked points are not trials");
    }

    #[test]
    fn an_empty_trial_still_cost_its_proposals() {
        let progress = Progress::empty().absorb(trial(0, 2_730));
        assert!(progress.is_empty());
        assert_eq!(progress.proposed(), 2_730);
        assert_eq!(progress.landed(), 0);
    }

    /// A million points absorbed leaves a window of the most recent
    /// thousand, oldest first.
    #[test]
    fn the_history_is_a_window_of_recent_points() {
        let mut progress = Progress::empty();
        for round in 0..1_000u32 {
            let points = (0..1_000u32)
                .map(|i| vec![f64::from(round * 1_000 + i)])
                .collect();
            progress = progress.extend(points);
        }
        assert_eq!(progress.points().len(), super::RECENT_POINTS);
        assert_eq!(progress.points().front(), Some(&vec![998_976.0]));
        assert_eq!(progress.points().back(), Some(&vec![999_999.0]));
    }
}
