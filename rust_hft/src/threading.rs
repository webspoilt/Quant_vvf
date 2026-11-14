//! Dedicated Core Thread Builder & CPU Affinity Manager

use core_affinity::{get_core_ids, set_for_current};
use std::thread::{Builder, JoinHandle};

#[allow(dead_code)]
pub struct ThreadAffinityConfig {
    pub thread_name: &'static str,
    pub core_index: usize,
}

pub struct ThreadBuilder;

impl ThreadBuilder {
    pub fn spawn_pinned<F, T>(
        name: &'static str,
        core_index: usize,
        work: F,
    ) -> Result<JoinHandle<T>, std::io::Error>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        Builder::new().name(name.to_string()).spawn(move || {
            if let Some(core_ids) = get_core_ids() {
                if let Some(target_core) = core_ids.get(core_index) {
                    let pinned = set_for_current(*target_core);
                    if pinned {
                        eprintln!(
                            "[ThreadBuilder] Thread '{}' successfully pinned to CPU Core #{}",
                            name, core_index
                        );
                    } else {
                        eprintln!(
                            "[ThreadBuilder] WARNING: Failed to pin thread '{}' to CPU Core #{}",
                            name, core_index
                        );
                    }
                } else {
                    eprintln!(
                        "[ThreadBuilder] WARNING: Core index {} exceeds available core count {}",
                        core_index,
                        core_ids.len()
                    );
                }
            } else {
                eprintln!(
                    "[ThreadBuilder] WARNING: Could not query core affinity on this platform"
                );
            }
            work()
        })
    }
}
