#! /usr/bin/env python3

import os
from pathlib import Path

from lab.environments import LocalEnvironment, BaselSlurmEnvironment

import common_setup
from common_setup import IssueConfig, IssueExperiment


REPO_DIR_LOCAL = Path("/home/aeneas/Git/downward-projects/downward").expanduser()
REPO_DIR_REMOTE = Path("/infai/meiaen00/downward").expanduser()
BENCHMARKS_DIR = os.environ["DOWNWARD_BENCHMARKS"]
BUILDS = ["release"]
REVISIONS = ["tiebreaking-reachability-neg"]
CONFIG_NICKS = [
    ("astar-hmax", ["--search", "astar(lmcut(pcfstrategy=hmax))"]),
    ("astar-hmaxtie", ["--search", "astar(lmcut(pcfstrategy=hmaxtie))"]),
]
CONFIGS = [
    IssueConfig(
        config_nick, config, build_options=[build], driver_options=["--build", build]
    )
    for build in BUILDS
    for config_nick, config in CONFIG_NICKS
]

SUITE = common_setup.DEFAULT_OPTIMAL_SUITE
REPO_DIR = REPO_DIR_REMOTE

ENVIRONMENT = BaselSlurmEnvironment(
    partition="infai_2",
    email="aeneas.meier@stud.unibas.ch",
    memory_per_cpu="3947M",
    export=["PATH"],
)

if common_setup.is_test_run():
    SUITE = IssueExperiment.DEFAULT_TEST_SUITE
    ENVIRONMENT = LocalEnvironment(processes=8)
    REPO_DIR = REPO_DIR_LOCAL

exp = IssueExperiment(
    REPO_DIR,
    revisions=REVISIONS,
    configs=CONFIGS,
    environment=ENVIRONMENT,
)
exp.add_suite(BENCHMARKS_DIR, SUITE)

exp.add_parser(exp.EXITCODE_PARSER)
exp.add_parser(exp.TRANSLATOR_PARSER)
exp.add_parser(exp.SINGLE_SEARCH_PARSER)
exp.add_parser(exp.PLANNER_PARSER)

exp.add_step("build", exp.build)
exp.add_step("start", exp.start_runs)
exp.add_step("parse", exp.parse)
exp.add_fetcher(name="fetch")

SPECIAL_ATTRIBUTES = []
ATTRIBUTES = exp.DEFAULT_TABLE_ATTRIBUTES + SPECIAL_ATTRIBUTES
SCATTER_ATTRIBUTES = SPECIAL_ATTRIBUTES

exp.add_absolute_report_step(attributes=ATTRIBUTES)
exp.add_comparison_table_step(attributes=ATTRIBUTES)
exp.add_scatter_plot_step(relative=False, attributes=["search_time", "expansions", "evaluations"])

exp.run_steps()
