//! Provides several common characteristic functions for
//! option pricing.  All of the characteristic functions
//! are with respect to "ui" instead of "u".

pub mod cgmy;
pub mod cir;
pub mod gamma;
pub mod gauss;
pub mod merton;
pub mod stable;
mod utils;
