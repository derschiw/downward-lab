# -*- coding: utf-8 -*-

# Last edited on May 14, 2025 to add IPC 2023 domains.

import itertools
import os
import platform
import re
import sys

from lab.experiment import ARGPARSER
from lab import tools

from downward.experiment import FastDownwardExperiment
from downward.reports.absolute import AbsoluteReport
from downward.reports.compare import ComparativeReport

from lib.plots import ScatterPlotReport


def parse_args():
    ARGPARSER.add_argument(
        "--test",
        choices=["yes", "no", "auto"],
        default="auto",
        dest="test_run",
        help="test experiment locally on a small suite if --test=yes or "
        "--test=auto and we are not on a cluster",
    )
    return ARGPARSER.parse_args()


ARGS = parse_args()


DEFAULT_OPTIMAL_SUITE = [
    "agricola-opt18-strips",
    "airport",
    "barman-opt11-strips",
    "barman-opt14-strips",
    "blocks",
    "childsnack-opt14-strips",
    "data-network-opt18-strips",
    "depot",
    "driverlog",
    "elevators-opt08-strips",
    "elevators-opt11-strips",
    "floortile-opt11-strips",
    "floortile-opt14-strips",
    "freecell",
    "ged-opt14-strips",
    "grid",
    "gripper",
    "hiking-opt14-strips",
    "logistics00",
    "logistics98",
    "miconic",
    "movie",
    "mprime",
    "mystery",
    "nomystery-opt11-strips",
    "openstacks-opt08-strips",
    "openstacks-opt11-strips",
    "openstacks-opt14-strips",
    "openstacks-strips",
    "organic-synthesis-opt18-strips",
    "organic-synthesis-split-opt18-strips",
    "parcprinter-08-strips",
    "parcprinter-opt11-strips",
    "parking-opt11-strips",
    "parking-opt14-strips",
    "pathways",
    "pegsol-08-strips",
    "pegsol-opt11-strips",
    "petri-net-alignment-opt18-strips",
    "pipesworld-notankage",
    "pipesworld-tankage",
    "psr-small",
    "quantum-layout-opt23-strips",
    "rovers",
    "satellite",
    "scanalyzer-08-strips",
    "scanalyzer-opt11-strips",
    "snake-opt18-strips",
    "sokoban-opt08-strips",
    "sokoban-opt11-strips",
    "spider-opt18-strips",
    "storage",
    "termes-opt18-strips",
    "tetris-opt14-strips",
    "tidybot-opt11-strips",
    "tidybot-opt14-strips",
    "tpp",
    "transport-opt08-strips",
    "transport-opt11-strips",
    "transport-opt14-strips",
    "trucks-strips",
    "visitall-opt11-strips",
    "visitall-opt14-strips",
    "woodworking-opt08-strips",
    "woodworking-opt11-strips",
    "zenotravel",
]

DEFAULT_SATISFICING_SUITE = [
    "agricola-sat18-strips",
    "airport",
    "assembly",
    "barman-sat11-strips",
    "barman-sat14-strips",
    "blocks",
    "caldera-sat18-adl",
    "caldera-split-sat18-adl",
    "cavediving-14-adl",
    "childsnack-sat14-strips",
    "citycar-sat14-adl",
    "data-network-sat18-strips",
    "depot",
    "driverlog",
    "elevators-sat08-strips",
    "elevators-sat11-strips",
    "flashfill-sat18-adl",
    "floortile-sat11-strips",
    "floortile-sat14-strips",
    "folding-sat23-adl",
    "freecell",
    "ged-sat14-strips",
    "grid",
    "gripper",
    "hiking-sat14-strips",
    "labyrinth-sat23-adl",
    "logistics00",
    "logistics98",
    "maintenance-sat14-adl",
    "miconic",
    "miconic-fulladl",
    "miconic-simpleadl",
    "movie",
    "mprime",
    "mystery",
    "nomystery-sat11-strips",
    "nurikabe-sat18-adl",
    "openstacks",
    "openstacks-sat08-adl",
    "openstacks-sat08-strips",
    "openstacks-sat11-strips",
    "openstacks-sat14-strips",
    "openstacks-strips",
    "organic-synthesis-sat18-strips",
    "organic-synthesis-split-sat18-strips",
    "parcprinter-08-strips",
    "parcprinter-sat11-strips",
    "parking-sat11-strips",
    "parking-sat14-strips",
    "pathways",
    "pegsol-08-strips",
    "pegsol-sat11-strips",
    "pipesworld-notankage",
    "pipesworld-tankage",
    "psr-small",
    "quantum-layout-sat23-strips",
    "recharging-robots-sat23-adl",
    "ricochet-robots-sat23-adl",
    "rovers",
    "rubiks-cube-sat23-adl",
    "satellite",
    "scanalyzer-08-strips",
    "scanalyzer-sat11-strips",
    "schedule",
    "settlers-sat18-adl",
    "slitherlink-sat23-adl",
    "snake-sat18-strips",
    "sokoban-sat08-strips",
    "sokoban-sat11-strips",
    "spider-sat18-strips",
    "storage",
    "termes-sat18-strips",
    "tetris-sat14-strips",
    "thoughtful-sat14-strips",
    "tidybot-sat11-strips",
    "tpp",
    "transport-sat08-strips",
    "transport-sat11-strips",
    "transport-sat14-strips",
    "trucks",
    "trucks-strips",
    "visitall-sat11-strips",
    "visitall-sat14-strips",
    "woodworking-sat08-strips",
    "woodworking-sat11-strips",
    "zenotravel",
]


def get_script():
    """Get file name of main script."""
    return tools.get_script_path()


def get_script_dir():
    """Get directory of main script.

    Usually a relative directory (depends on how it was called by the user.)"""
    return os.path.dirname(get_script())


def get_experiment_name():
    """Get name for experiment.

    Derived from the absolute filename of the main script, e.g.
    "/ham/spam/eggs.py" => "spam-eggs"."""
    script = os.path.abspath(get_script())
    script_dir = os.path.basename(os.path.dirname(script))
    script_base = os.path.splitext(os.path.basename(script))[0]
    return "%s-%s" % (script_dir, script_base)


def get_data_dir():
    """Get data dir for the experiment.

    This is the subdirectory "data" of the directory containing
    the main script."""
    return os.path.join(get_script_dir(), "data", get_experiment_name())


def get_repo_base():
    """Get base directory of the repository, as an absolute path.

    Search upwards in the directory tree from the main script until a
    directory with a subdirectory named ".git" is found.

    Abort if the repo base cannot be found."""
    path = os.path.abspath(get_script_dir())
    while os.path.dirname(path) != path:
        if os.path.exists(os.path.join(path, ".git")):
            return path
        path = os.path.dirname(path)
    sys.exit("repo base could not be found")


def is_running_on_cluster():
    return re.fullmatch(r"login12|ic[ab]\d\d", platform.node())


def is_test_run():
    return ARGS.test_run == "yes" or (
        ARGS.test_run == "auto" and not is_running_on_cluster()
    )


def get_algo_nick(revision, config_nick):
    return f"{revision}-{config_nick}"


class IssueConfig(object):
    """Hold information about a planner configuration.

    See FastDownwardExperiment.add_algorithm() for documentation of the
    constructor's options.

    """

    def __init__(
        self, nick, component_options, build_options=None, driver_options=None
    ):
        self.nick = nick
        self.component_options = component_options
        self.build_options = build_options
        self.driver_options = driver_options


class IssueExperiment(FastDownwardExperiment):
    """Subclass of FastDownwardExperiment with some convenience features."""

    DEFAULT_TEST_SUITE = [
        "depot:p01.pddl",
        "gripper:prob01.pddl",
        "gripper:prob02.pddl",
        # "gripper:prob03.pddl",
        # "gripper:prob04.pddl",
        "blocks:probBLOCKS-4-0.pddl",
        "blocks:probBLOCKS-6-0.pddl",
        # "blocks:probBLOCKS-8-0.pddl",
        # "blocks:probBLOCKS-14-1.pddl",
        "transport-opt08-strips:p01.pddl",
        "transport-opt08-strips:p02.pddl",
        # "transport-opt08-strips:p03.pddl",
        "zenotravel:p01.pddl",
        "zenotravel:p02.pddl",
        # "zenotravel:p03.pddl",
        # "zenotravel:p04.pddl",
        "satellite:p01-pfile1.pddl",
        "scanalyzer-08-strips:p01.pddl",
        # "elevators-opt08-strips:p03.pddl",
        # "elevators-opt08-strips:p04.pddl",
        "storage:p01.pddl",
        "storage:p02.pddl",
        # "storage:p03.pddl",
        # "storage:p04.pddl",
        # "spider-opt18-strips:p03.pddl",
        # "spider-opt18-strips:p04.pddl",
        "sokoban-opt08-strips:p01.pddl",
        "woodworking-opt08-strips:p01.pddl",
    ]

    DEFAULT_TABLE_ATTRIBUTES = [
        "cost",
        "coverage",
        "error",
        "evaluations",
        "expansions",
        "expansions_until_last_jump",
        "initial_h_value",
        "generated",
        "memory",
        "planner_memory",
        "planner_time",
        "quality",
        "run_dir",
        "score_evaluations",
        "score_expansions",
        "score_generated",
        "score_memory",
        "score_search_time",
        "score_total_time",
        "search_time",
        "total_time",
    ]

    DEFAULT_SCATTER_PLOT_ATTRIBUTES = [
        "evaluations",
        "expansions",
        "expansions_until_last_jump",
        "initial_h_value",
        "memory",
        "search_time",
        "total_time",
    ]

    PORTFOLIO_ATTRIBUTES = [
        "cost",
        "coverage",
        "error",
        "plan_length",
        "run_dir",
    ]

    def __init__(
        self, repo_path=None, revisions=None, configs=None, path=None, **kwargs
    ):
        """

        You can either specify both *revisions* and *configs* or none
        of them. If they are omitted, you will need to call
        exp.add_algorithm() manually.

        If *revisions* is given, it must be a non-empty list of
        revision identifiers, which specify which planner versions to
        use in the experiment. The same versions are used for
        translator, preprocessor and search. ::

            IssueExperiment(revisions=["issue123", "4b3d581643"], ...)

        If *configs* is given, it must be a non-empty list of
        IssueConfig objects. ::

            IssueExperiment(..., configs=[
                IssueConfig("ff", ["--search", "eager_greedy(ff())"]),
                IssueConfig(
                    "lama", [],
                    driver_options=["--alias", "seq-sat-lama-2011"]),
            ])

        If *path* is specified, it must be the path to where the
        experiment should be built (e.g.
        /home/john/experiments/issue123/exp01/). If omitted, the
        experiment path is derived automatically from the main
        script's filename. Example::

            script = experiments/issue123/exp01.py -->
            path = experiments/issue123/data/issue123-exp01/

        """

        path = path or get_data_dir()

        FastDownwardExperiment.__init__(self, path=path, **kwargs)

        if repo_path is None:
            repo_path = get_repo_base()

        if (revisions and not configs) or (not revisions and configs):
            raise ValueError(
                "please provide either both or none of revisions and configs"
            )

        if all(isinstance(rev, tuple) for rev in revisions):
            pass
        else:
            revisions = [(rev, rev) for rev in revisions]

        for rev, rev_nick in revisions:
            for config in configs:
                self.add_algorithm(
                    get_algo_nick(rev_nick, config.nick),
                    repo_path,
                    rev,
                    config.component_options,
                    build_options=config.build_options,
                    driver_options=config.driver_options,
                )

        self._revisions = [rev[0] for rev in revisions]
        self._configs = configs

    @classmethod
    def _is_portfolio(cls, config_nick):
        return "fdss" in config_nick

    @classmethod
    def get_supported_attributes(cls, config_nick, attributes):
        if cls._is_portfolio(config_nick):
            return [attr for attr in attributes if attr in cls.PORTFOLIO_ATTRIBUTES]
        return attributes

    def add_absolute_report_step(self, **kwargs):
        """Add step that makes an absolute report.

        Absolute reports are useful for experiments that don't compare
        revisions.

        The report is written to the experiment evaluation directory.

        All *kwargs* will be passed to the AbsoluteReport class. If the
        keyword argument *attributes* is not specified, a default list
        of attributes is used. ::

            exp.add_absolute_report_step(attributes=["coverage"])

        """
        kwargs.setdefault("attributes", self.DEFAULT_TABLE_ATTRIBUTES)
        report = AbsoluteReport(**kwargs)
        outfile = os.path.join(
            self.eval_dir, get_experiment_name() + "." + report.output_format
        )
        self.add_report(report, outfile=outfile)

    def add_comparison_table_step(self, revision_pairs=[], **kwargs):
        """Add a step that makes pairwise revision comparisons.

        Create comparative reports for all pairs of Fast Downward
        revisions. Each report pairs up the runs of the same config and
        lists the two absolute attribute values and their difference
        for all attributes in kwargs["attributes"].

        All *kwargs* will be passed to the CompareConfigsReport class.
        If the keyword argument *attributes* is not specified, a
        default list of attributes is used. ::

            exp.add_comparison_table_step(attributes=["coverage"])

        """
        kwargs.setdefault("attributes", self.DEFAULT_TABLE_ATTRIBUTES)

        if not revision_pairs:
            revision_pairs = [
                (rev1, rev2)
                for rev1, rev2 in itertools.combinations(self._revisions, 2)
            ]

        def make_comparison_tables():
            for rev1, rev2 in revision_pairs:
                compared_configs = []
                for config in self._configs:
                    config_nick = config.nick
                    compared_configs.append(
                        (
                            "%s-%s" % (rev1, config_nick),
                            "%s-%s" % (rev2, config_nick),
                            "Diff (%s)" % config_nick,
                        )
                    )
                report = ComparativeReport(compared_configs, **kwargs)
                outfile = os.path.join(
                    self.eval_dir,
                    "%s-%s-%s-compare.%s"
                    % (self.name, rev1, rev2, report.output_format),
                )
                report(self.eval_dir, outfile)

        self.add_step("make-comparison-tables", make_comparison_tables)

    def add_scatter_plot_step(self, relative=False, attributes=None, additional=[]):
        """Add step creating (relative) scatter plots for all revision pairs.

        Create a scatter plot for each combination of attribute,
        configuration and revisions pair. If *attributes* is not
        specified, a list of common scatter plot attributes is used.
        For portfolios all attributes except "cost", "coverage" and
        "plan_length" will be ignored. ::

            exp.add_scatter_plot_step(attributes=["expansions"])

        """
        if relative:
            scatter_dir = os.path.join(self.eval_dir, "scatter-relative")
            step_name = "make-relative-scatter-plots"
        else:
            scatter_dir = os.path.join(self.eval_dir, "scatter-absolute")
            step_name = "make-absolute-scatter-plots"
        if attributes is None:
            attributes = self.DEFAULT_SCATTER_PLOT_ATTRIBUTES

        def make_scatter_plot(config_nick, rev1, rev2, attribute, config_nick2=None):
            name = "-".join([self.name, rev1, rev2, attribute, config_nick])
            if config_nick2 is not None:
                name += "-" + config_nick2
            algo1 = get_algo_nick(rev1, config_nick)
            algo2 = get_algo_nick(
                rev2, config_nick if config_nick2 is None else config_nick2
            )
            report = ScatterPlotReport(
                filter_algorithm=[algo1, algo2],
                attributes=[attribute],
                relative=relative,
                get_category=lambda run1, run2: run1["domain"],
            )
            report(self.eval_dir, os.path.join(scatter_dir, rev1 + "-" + rev2, name))

        def make_scatter_plots():
            for config in self._configs:
                for rev1, rev2 in itertools.combinations(self._revisions, 2):
                    for attribute in self.get_supported_attributes(
                        config.nick, attributes
                    ):
                        make_scatter_plot(config.nick, rev1, rev2, attribute)
            for nick1, nick2, rev1, rev2, attribute in additional:
                make_scatter_plot(nick1, rev1, rev2, attribute, config_nick2=nick2)

        self.add_step(step_name, make_scatter_plots)

    def add_config_based_scatter_plot_step(
        self, relative=False, attributes=None, additional=[]
    ):
        """Add step creating (relative) scatter plots for all revision pairs.

        Create a scatter plot for each combination of attribute,
        configuration and revisions pair. If *attributes* is not
        specified, a list of common scatter plot attributes is used.
        For portfolios all attributes except "cost", "coverage" and
        "plan_length" will be ignored. ::

            exp.add_config_based_scatter_plot_step(attributes=["expansions"])

        """
        matplotlib_options = {
            "font.family": "serif",
            "font.weight": "normal",
            # Used if more specific sizes not set.
            "font.size": 80,
            "axes.labelsize": 20,
            "axes.titlesize": 30,
            "legend.fontsize": 22,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.markersize": 20,
            "lines.markeredgewidth": 0.25,
            "lines.linewidth": 1,
            # Width and height in inches.
            "figure.figsize": [18, 18],
            "savefig.dpi": 100,
        }
        if relative:
            scatter_dir = os.path.join(self.eval_dir, "scatter-relative")
            step_name = "make-config-based-relative-scatter-plots"
        else:
            scatter_dir = os.path.join(self.eval_dir, "scatter-absolute")
            step_name = "make-config-based-absolute-scatter-plots"
        if attributes is None:
            attributes = self.DEFAULT_SCATTER_PLOT_ATTRIBUTES

        def make_scatter_plot(config_nick, rev1, rev2, attribute):
            name = "-".join([self.name, config_nick, rev1, "vs", rev2, attribute])

            algo1 = f"{config_nick}-{rev1}"
            algo2 = f"{config_nick}-{rev2}"

            report = ScatterPlotReport(
                filter_algorithm=[algo1, algo2],
                attributes=[attribute],
                relative=relative,
                get_category=lambda run1, run2: run1["domain"],
                matplotlib_options=matplotlib_options,
                format="png",
                title="",
                xlabel="FD Tiebreaking",
                ylabel="Reachability",
            )
            report(self.eval_dir, os.path.join(scatter_dir, rev1 + "-" + rev2, name))

        def make_scatter_plots():
            for rev in self._revisions:
                print(rev)
                for config1, config2 in itertools.combinations(self._configs, 2):
                    for attribute in self.get_supported_attributes(rev, attributes):
                        make_scatter_plot(rev, config1.nick, config2.nick, attribute)

        self.add_step(step_name, make_scatter_plots)

    def add_cost_vs_initial_h_value_comparison_table_step(self, relative=False):
        """Add step that creates comparison tables for initial_h_value and cost between all configurations.

        Creates two tables comparing initial_h_value and cost values across all configuration pairs
        for each revision.
        """

        def extract_all_tables(html_file, csv_file, table_container_id):
            """
            Extract all tables that start with the given prefix and save them to CSV.
            """
            with open(html_file, "r") as f:
                html_content = f.read()

            # Find all tables with IDs starting with the given prefix
            table_pattern = re.compile(
                rf'<section id="({table_container_id}[^"]*?)">(.+?)</section>',
                re.DOTALL,
            )
            matches = table_pattern.findall(html_content)

            if not matches:
                print(
                    f"No tables with id prefix '{table_container_id}' found in {html_file}."
                )
                return

            from bs4 import BeautifulSoup
            import csv

            for table_id, table_html in matches:
                print(f"Extracting table: {table_id}")

                soup = BeautifulSoup(table_html, "html.parser")
                table = soup.find("table")

                if table:
                    # Create a unique CSV filename for each table
                    table_csv_file = csv_file.replace(".csv", f"-{table_id}.csv")

                    with open(table_csv_file, "w", newline="") as csvfile:
                        csvwriter = csv.writer(csvfile)
                        for row in table.find_all("tr"):
                            cols = row.find_all(["td", "th"])
                            csvwriter.writerow(
                                [col.get_text(strip=True) for col in cols]
                            )

                    print(f"Saved table {table_id} to {table_csv_file}")
                else:
                    print(f"No table found in section {table_id}")

        def create_summary_table(input_dir, outfile):
            """
            Does the following:
            - Loop over all CSV starting with 'cost_' in the input_dir
                - Loop over all rows with values for '.pddl' files (skip header and footer)
                    - Ignore the line if cost is none
                    - Extract the corresponding line from the initial_h_value CSV file else
                - Write a summary CSV with columns: problem, cost_rev1, cost_rev2, cost_diff, initial_h_value_rev1, initial_h_value_rev2, initial_h_value_diff
            """
            import csv
            import os
            import glob

            # Find all CSV files starting with 'cost_'
            cost_files = glob.glob(os.path.join(input_dir, "cost_*.csv"))

            if not cost_files:
                print(f"No cost CSV files found in {input_dir}")
                return

            summary_data = []

            for cost_file in cost_files:
                print(f"Processing cost file: {cost_file}")

                # Determine corresponding initial_h_value file
                cost_basename = os.path.basename(cost_file)

                # Try different naming patterns to find the matching initial_h_value file
                possible_names = []

                if "-cost-compare-cost-" in cost_basename:
                    # Pattern: cost_...-cost-compare-cost-domain.csv -> initial_h_value_...-initial-h-value-compare-initial_h_value-domain.csv
                    possible_names.append(
                        cost_basename.replace("cost_", "initial_h_value_").replace(
                            "-cost-compare-cost-",
                            "-initial-h-value-compare-initial_h_value-",
                        )
                    )
                elif "-cost-compare--" in cost_basename:
                    # Try multiple patterns for double-dash cost files
                    # Pattern 1: cost_...-cost-compare--domain.csv -> initial_h_value_...-initial-h-value-compare--domain.csv (same pattern)
                    possible_names.append(
                        cost_basename.replace("cost_", "initial_h_value_").replace(
                            "-cost-compare--", "-initial-h-value-compare--"
                        )
                    )
                    # Pattern 2: cost_...-cost-compare--domain.csv -> initial_h_value_...-initial-h-value-compare-initial_h_value-domain.csv (cross-pattern)
                    possible_names.append(
                        cost_basename.replace("cost_", "initial_h_value_").replace(
                            "-cost-compare--",
                            "-initial-h-value-compare-initial_h_value-",
                        )
                    )
                else:
                    # Fallback: simple replacement
                    possible_names.append(
                        cost_basename.replace("cost_", "initial_h_value_")
                    )  # Find the first matching file
                initial_h_file = None
                for name in possible_names:
                    candidate_file = os.path.join(input_dir, name)
                    if os.path.exists(candidate_file):
                        initial_h_file = candidate_file
                        break

                if initial_h_file is None:
                    # If no direct match found, use the first candidate for error reporting
                    initial_h_file = os.path.join(input_dir, possible_names[0])

                if not os.path.exists(initial_h_file):
                    print(
                        f"Warning: Corresponding initial_h_value file not found: {initial_h_file}"
                    )
                    continue

                # Read cost data
                cost_data = {}
                with open(cost_file, "r") as f:
                    reader = csv.reader(f)
                    rows = list(reader)

                    if len(rows) < 2:
                        continue

                    # Skip header and footer, process data rows
                    for i, row in enumerate(rows[1:], 1):  # Skip header
                        if len(row) < 4:  # Need at least problem, cost1, cost2, diff
                            continue

                        problem = row[0]
                        if not problem.endswith(".pddl"):
                            continue

                        try:
                            cost1 = row[1] if row[1] and row[1] != "None" else None
                            cost2 = row[2] if row[2] and row[2] != "None" else None
                            cost_diff = row[3] if row[3] and row[3] != "None" else None

                            # Skip if both costs are None
                            if cost1 is None and cost2 is None:
                                continue

                            cost_data[problem] = {
                                "cost1": cost1,
                                "cost2": cost2,
                                "cost_diff": cost_diff,
                            }
                        except (ValueError, IndexError):
                            continue

                # Read initial_h_value data
                initial_h_data = {}
                with open(initial_h_file, "r") as f:
                    reader = csv.reader(f)
                    rows = list(reader)

                    if len(rows) < 2:
                        continue

                    for i, row in enumerate(rows[1:], 1):  # Skip header
                        if len(row) < 4:
                            continue

                        problem = row[0]
                        if not problem.endswith(".pddl"):
                            continue

                        try:
                            h1 = row[1] if row[1] and row[1] != "None" else None
                            h2 = row[2] if row[2] and row[2] != "None" else None
                            h_diff = row[3] if row[3] and row[3] != "None" else None

                            initial_h_data[problem] = {
                                "h1": h1,
                                "h2": h2,
                                "h_diff": h_diff,
                            }
                        except (ValueError, IndexError):
                            continue

                # Combine data for problems that have cost data
                for problem in cost_data:
                    cost_info = cost_data[problem]
                    h_info = initial_h_data.get(
                        problem, {"h1": None, "h2": None, "h_diff": None}
                    )

                    summary_data.append(
                        {
                            "problem": problem,
                            "cost_rev1": cost_info["cost1"],
                            "cost_rev2": cost_info["cost2"],
                            "cost_diff": cost_info["cost_diff"],
                            "initial_h_value_rev1": h_info["h1"],
                            "initial_h_value_rev2": h_info["h2"],
                            "initial_h_value_diff": h_info["h_diff"],
                        }
                    )

            # Write summary CSV
            if summary_data:
                with open(outfile, "w", newline="") as f:
                    fieldnames = [
                        "problem",
                        "cost_rev1",
                        "cost_rev2",
                        "cost_diff",
                        "initial_h_value_rev1",
                        "initial_h_value_rev2",
                        "initial_h_value_diff",
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(summary_data)

                print(
                    f"Summary table written to {outfile} with {len(summary_data)} rows"
                )
            else:
                print("No data found to create summary table")

        def make_comparison_tables():
            for rev in self._revisions:
                # Create algorithm pairs for all configuration combinations
                algorithm_pairs = []
                # for config1, config2 in itertools.combinations(self._configs, 2):
                #     algo1 = get_algo_nick(rev, config1.nick)
                #     algo2 = get_algo_nick(rev, config2.nick)
                #     diff_label = f"Diff {config1.nick}-{config2.nick}"
                #     algorithm_pairs.append((algo1, algo2, diff_label))

                # the upper code is correct, but we only care about hmax vs hadd
                algorithm_pairs = [
                    ("main-astar-hmax", "main-astar-hadd", "Diff hmax-hadd")
                ]

                # Create initial_h_value comparison table
                initial_h_value_report = ComparativeReport(
                    algorithm_pairs, attributes=["initial_h_value"]
                )
                initial_h_value_outfile = os.path.join(
                    self.eval_dir,
                    f"{self.name}-{rev}-initial-h-value-compare.{initial_h_value_report.output_format}",
                )
                initial_h_value_report(self.eval_dir, initial_h_value_outfile)

                # Create cost comparison table
                cost_report = ComparativeReport(algorithm_pairs, attributes=["cost"])
                print("cost report", cost_report)
                cost_outfile = os.path.join(
                    self.eval_dir,
                    f"{self.name}-{rev}-cost-compare.{cost_report.output_format}",
                )
                cost_report(self.eval_dir, cost_outfile)

                csv_dir = os.path.join(self.eval_dir, "csv")
                os.makedirs(csv_dir, exist_ok=True)
                csv_filename = os.path.basename(cost_outfile.replace(".html", ".csv"))
                extract_all_tables(
                    cost_outfile,
                    os.path.join(csv_dir, f"cost_{csv_filename}"),
                    "cost",
                )

                csv_filename = os.path.basename(
                    initial_h_value_outfile.replace(".html", ".csv")
                )
                extract_all_tables(
                    initial_h_value_outfile,
                    os.path.join(csv_dir, f"initial_h_value_{csv_filename}"),
                    "initial_h_value",
                )
                # Create summary table
                summary_outfile = os.path.join(
                    csv_dir, f"summary_{rev}_cost_vs_initial_h.csv"
                )
                create_summary_table(csv_dir, summary_outfile)

        self.add_step(
            "make-cost-vs-initial-h-comparison-tables", make_comparison_tables
        )

    def add_archive_step(self, archive_path):
        archive.add_archive_step(self, archive_path)

    def add_archive_eval_dir_step(self, archive_path):
        archive.add_archive_eval_dir_step(self, archive_path)
