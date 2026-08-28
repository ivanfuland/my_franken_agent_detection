//! Drill asset (fad-fork fix/codex-legacy-token-count-shape) -- not a
//! product feature, not wired into any command surface.
//!
//! Read-only sweep of a real codex corpus root (a locally mounted
//! historical archive, passed as the `--scan-root` CLI argument) using
//! the actual codex connector's
//! `discover_source_files` + per-file `scan()`, to confirm that the
//! `rate_limits`-optional fix (this branch, commit 42780aa) is sufficient:
//! that the 7 known files (101 token_count events, 2025-09-17, missing
//! `rate_limits`) are the *only* legacy shape lurking in this corpus,
//! not just the first one `cass ingest manifest`'s all-or-nothing
//! `scan_with_callback` happened to trip on.
//!
//! Scans each discovered rollout file individually (not one
//! `scan_with_callback` call over the whole root) specifically so one
//! file's error does not stop enumeration of the rest -- this is the one
//! property `cass ingest manifest` itself lacks, and exactly why manifest
//! generation on this corpus aborted after the first bad file instead of reporting
//! all of them.
//!
//! Reads only; writes only to the given `--out` path (outside the corpus root being
//! scanned).

use std::env;
use std::path::PathBuf;

use franken_agent_detection::CodexConnector;
use franken_agent_detection::connectors::{Connector, DiscoveredSourceRole, ScanContext, ScanRoot};
use serde::Serialize;

#[derive(Serialize)]
struct FileResult {
    path: String,
    ok: bool,
    conversations: usize,
    error: Option<String>,
}

#[derive(Serialize)]
struct Report {
    scan_root: String,
    discovered_files: usize,
    ok_files: usize,
    failed_files: usize,
    failures: Vec<FileResult>,
}

fn main() {
    let mut args = env::args().skip(1);
    let mut scan_root: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--scan-root" => scan_root = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            other => panic!("unknown arg: {other}"),
        }
    }
    let scan_root = scan_root.expect("--scan-root required");
    let out = out.expect("--out required");

    let connector = CodexConnector::new();
    let discover_ctx = ScanContext::with_roots(
        PathBuf::new(),
        vec![ScanRoot::local(scan_root.clone())],
        None,
    );
    let discovered = connector
        .discover_source_files(&discover_ctx)
        .expect("discover_source_files failed");

    let mut primary_paths: Vec<PathBuf> = discovered
        .into_iter()
        .filter(|f| f.role == DiscoveredSourceRole::PrimarySessionLog)
        .map(|f| f.source_path)
        .collect();
    primary_paths.sort();
    primary_paths.dedup();

    let mut ok_files = 0usize;
    let mut failures = Vec::new();

    for path in &primary_paths {
        let per_file_ctx = ScanContext::with_roots(
            path.parent().unwrap_or(&scan_root).to_path_buf(),
            vec![ScanRoot::local(path.clone())],
            None,
        );
        match connector.scan(&per_file_ctx) {
            Ok(convs) => {
                ok_files += 1;
                let _ = convs; // conversations count not needed on success
            }
            Err(e) => {
                failures.push(FileResult {
                    path: path.display().to_string(),
                    ok: false,
                    conversations: 0,
                    error: Some(format!("{e:#}")),
                });
            }
        }
    }

    let report = Report {
        scan_root: scan_root.display().to_string(),
        discovered_files: primary_paths.len(),
        ok_files,
        failed_files: failures.len(),
        failures,
    };

    let json = serde_json::to_string_pretty(&report).unwrap();
    std::fs::write(&out, &json).expect("failed to write report");
    println!("{json}");
    eprintln!(
        "\n[fad_codex_corpus_shape_probe] discovered={} ok={} failed={}",
        report.discovered_files, report.ok_files, report.failed_files
    );
}
