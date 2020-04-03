#!/usr/bin/env python
from __future__ import print_function, division

import traceback

from subprocess import Popen, PIPE

import sys
import os
import os.path
import shlex
import sqlite3
import time
import json
import csv
import math

from datetime import datetime

import logging
logging.basicConfig(level=logging.DEBUG)

from argparse import ArgumentParser

from contextlib import contextmanager


# copy current env, we will modify and use for external commands
env = os.environ.copy()


benchmark_dir = os.path.abspath(os.path.dirname(__file__))


#
# default configuration options
#

conf = {
    # directories used during benchmarking
    "working_dir": benchmark_dir,
    "repo_dir":    "repos",
    "logs_dir":    "logs",

    # remove benchmark data file after adding to database
    "remove_csv":  False,

    # generate a plot showing history of each benchmark
    "plot_history": True,

    # CA cert information needed by curl (defaults)
    "no-ca": {
        "cacert":  "/etc/ssl/certs/ca-certificates.crt",
        "capath":  "/etc/ssl/certs"
    },
}


#
# utility functions
#

def read_json(filename):
    """
    read data from a JSON file
    """
    with open(filename) as json_file:
        return json.loads(json_file.read())


def write_json(filename, data):
    """
    write JSON data to a file
    """
    with open(filename, 'w') as json_file:
        json_file.write(json.dumps(data))


def prepend_path(newdir, path):
    """
    prepend `newdir` to `path`, return modified `path`
    """
    dirs = path.split(os.pathsep)
    dirs.insert(0, newdir)
    path = (os.pathsep).join(dirs)
    return path


def init_env(project_info):
    """
    initialize env for commands with benchmark and project customizations
    """
    # reset env to be copy of current OS env
    global env
    env = os.environ.copy()

    # prepend benchmark dir to PATH
    env["PATH"] = prepend_path(benchmark_dir, env["PATH"])

    # add any env vars from benchmark config
    if "env" in conf:
        for key, val in conf["env"].items():
            if val.find('~') >= 0:
                val = os.path.expanduser(val)
            if val.find('$') >= 0:
                val = os.path.expandvars(val)
            val = val.replace("$PYTHONPATH", "")  # in case it was empty
            print("setting benchmark ENV:", key, "=", val)
            env[key] = val

    # add any project specific env vars
    if "env" in project_info:
        for key, val in project_info["env"].items():
            if val.find('~') >= 0:
                val = os.path.expanduser(val)
            if val.find('$') >= 0:
                val = os.path.expandvars(val)
            val = val.replace("$PYTHONPATH", "")  # in case it was empty
            print("setting %s ENV:" % project_info["name"], key, "=", val)
            env[key] = val


def execute_cmd(cmd):
    """
    Execute the external command and get its exitcode, stdout and stderr.
    """
    logging.info("> %s", cmd)
    args = shlex.split(cmd)
    proc = Popen(args, stdout=PIPE, stderr=PIPE, env=env, universal_newlines=True)
    out, err = proc.communicate()

    rc = proc.returncode
    if rc:
        logging.info("RC: %d", rc)
    if out.strip():
        logging.debug(out)
    if rc and err.strip():  # disregard inconsequential stderr output
        logging.debug("STDERR:\n%s", err)

    return rc, out, err


def remove_dir(dirname):
    """
    Remove repo directory before a benchmarking run.
    Will force fresh cloning and avoid branch issues.
    """
    remove_cmd = "rm -rf " + dirname

    if os.path.exists(dirname):
        code, out, err = execute_cmd(remove_cmd)


def upload(files, dest):
    """
    upload files to destination via scp
    """
    cmd = "scp %s %s" % (" ".join(files), dest)
    code, out, err = execute_cmd(cmd)
    return code


def init_logging():
    """
    initialize logging to stdout with clean format
    """
    log = logging.getLogger()

    # remove old handler(s)
    for hdlr in log.handlers:
        log.removeHandler(hdlr)

    # set stdout handler with clean format
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)


def init_log_file(name):
    """
    initialize log file with given name
    """
    logs_dir = os.path.expanduser(os.path.join(conf["logs_dir"]))
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    filename = os.path.join(logs_dir, name+".log")
    fh = logging.FileHandler(filename)

    format_str = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    fh.formatter = logging.Formatter(format_str)

    log = logging.getLogger()
    log.addHandler(fh)


def close_log_file():
    """
    close current log file(s)
    """
    log = logging.getLogger()
    handlers = log.handlers[:]
    for handler in handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            log.removeHandler(handler)


#
# context managers
#

@contextmanager
def cd(newdir):
    """
    A cd that will better handle error and return to its orig dir.
    """
    logging.info('> cd %s', newdir)
    prevdir = os.getcwd()
    fulldir = os.path.expanduser(newdir)
    if not os.path.exists(fulldir):
        os.makedirs(fulldir)
    os.chdir(fulldir)
    try:
        yield
    finally:
        logging.info('> cd %s (from %s)', prevdir, fulldir)
        os.chdir(prevdir)


@contextmanager
def repo(repository, branch=None):
    """
    cd into local copy of repository.  if the repository has not been
    cloned yet, then clone it to working directory first.
    """
    prev_dir = os.getcwd()

    repo_dir = conf["repo_dir"]
    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir)
    logging.info('> cd %s (from %s)', repo_dir, prev_dir)
    os.chdir(repo_dir)

    repo_name = repository.split('/')[-1]
    if not os.path.isdir(repo_name):
        clone_repo(repository, branch)

    logging.info('> cd %s (REPO)', repo_name)
    os.chdir(repo_name)
    try:
        yield
    finally:
        logging.info('> cd %s (from REPO %s)', prev_dir, repo_name)
        os.chdir(prev_dir)


@contextmanager
def conda(conda):
    """
    activate conda environment (switching out global env).
    """
    global env
    save_env = env
    env = conda.env

    logging.info('> switching to Conda environment %s', conda.name)
    try:
        yield
    finally:
        logging.info('> restoring environment (from %s)', conda.name)
        env = save_env

#
# repository helpers
#

repo_types = {}


def clone_repo(repository, branch):
    """
    clone repository into current directory
    """
    if branch:
        git_clone_cmd = "git clone -b %s --single-branch %s" % (branch, repository)
        hg_clone_cmd = "hg clone %s -r %s" % (repository, branch)
    else:
        git_clone_cmd = "git clone " + repository
        hg_clone_cmd = "hg clone " + repository

    repo_type = repo_types.get(repository)
    if repo_type is "git":
        code, out, err = execute_cmd(git_clone_cmd)
    elif repo_type is "hg":
        code, out, err = execute_cmd(hg_clone_cmd)
    else:
        code, out, err = execute_cmd(git_clone_cmd)
        if not code:
            repo_types[repository] = "git"
        else:
            code, out, err = execute_cmd(hg_clone_cmd)
            if not code:
                repo_types[repository] = "hg"
    if code:
        raise RuntimeError("Could not clone %s" % repository)


def get_current_commit(repository):
    """
    Update and check the current repo for the most recent commit.
    """
    git_pull = "git pull"
    git_commit = "git rev-parse HEAD"

    hg_pull = "hg pull"
    hg_merge = "hg merge"
    hg_commit = "hg id -i"

    # pull latest commit from desired branch and get the commit ID
    repo_type = repo_types.get(repository)
    if repo_type is "git":
        code, out, err = execute_cmd(git_pull)
        code, out, err = execute_cmd(git_commit)
    elif repo_type is "hg":
        code, out, err = execute_cmd(hg_pull)
        code, out, err = execute_cmd(hg_merge)
        code, out, err = execute_cmd(hg_commit)
    else:
        code, out, err = execute_cmd(hg_pull)
        if (code is 0):
            code, out, err = execute_cmd(hg_merge)
            code, out, err = execute_cmd(hg_commit)
        else:
            code, out, err = execute_cmd(git_pull)
            code, out, err = execute_cmd(git_commit)

    return out.strip()   # TODO: strip the old commit IDs in the database as well


#
# worker classes
#


class CondaEnv(object):
    """
    this class encapsulates the logic required to create a conda environment
    """

    def __init__(self, name, dependencies, local_repos):
        """
        Create conda env, install dependencies .and then any local repositories.
        """
        self.name = name

        logging.info("============= CREATE ENV =============")

        cmd = "conda create -y -q -n " + name

        # handle python and numpy/scipy dependencies
        for dep in dependencies:
            if dep.startswith("python") or dep.startswith("numpy") or dep.startswith("scipy"):
                cmd = cmd + " " + dep

        # add other required packages
        conda_pkgs = " ".join([
            "git",              # for cloning git repos
            "pip<20.0",         # for installing dependencies
            "swig",             # for building dependencies
            "cython",           # for building dependencies
            "psutil",           # for testflo benchmarking
            # "memory_profiler",  # for testflo benchmarking
            "nomkl",            # TODO: experiment with this
            "matplotlib",       # for plotting results
            "curl",             # for uploading files & slack messages
            "sqlite"            # for backing up the database
        ])
        cmd = cmd + " " + conda_pkgs

        code, out, err = execute_cmd(cmd)
        if (code != 0):
            raise RuntimeError("Failed to create conda environment", name, code, out, err)

        # modify PATH for environment
        path = env["PATH"].split(os.pathsep)
        for dirname in path:
            if "anaconda" in dirname or "miniconda" in dirname:
                conda_dir = dirname
                path.remove(conda_dir)
                break
        
        self.env = env.copy()
        self.env["ORIG_CONDA_DIR"] = conda_dir

        self.env_path = conda_dir.replace("bin", "envs/"+name)
        self.python = self.env_path + "/bin/python"

        self.env["PATH"] = prepend_path(self.env_path+"/bin", (os.pathsep).join(path))
        logging.info("env_name: %s, path: %s" % (name, self.env["PATH"]))

        # install dependencies
        for dependency in dependencies:
            # install the proper version of testflo to do the benchmarking
            if dependency.startswith("python=3"):
                self.install("testflo", options="")
            elif dependency.startswith("python=2"):
                self.install("testflo<1.4", options="")

            logging.info("Installing dependency: %s" % dependency)
            if dependency.startswith("~") or dependency.startswith("/"):
                with cd(os.path.expanduser(dependency)):
                    if os.path.exists('requirements.txt'):
                        self.install("requirements.txt", options="-r")
                    self.install(".", options="")
            # python, numpy and scipy are installed when the env is created
            elif (not dependency.startswith("python=") and
                  not dependency.startswith("numpy") and not dependency.startswith("scipy")):
                self.install(dependency)

        # install from local repos
        for local_repo in local_repos:
            logging.info("Installing from local repo: %s" % local_repo)
            with repo(local_repo):
                if os.path.exists('requirements.txt'):
                    self.install("requirements.txt", options="-r")
                self.install(".")

    def install(self, package, extras="", options="-q"):
        """
        Install a package.
        """
        pipinstall = "%s -m pip install %s %s%s " % (self.python, options, package, extras)

        code, out, err = execute_cmd(pipinstall)

        if (code != 0) and package == ".":
            logging.info("pip install failed, trying with 'python setup.py'")
            # need to install with --prefix to get things installed into proper conda env
            code, out, err = execute_cmd("python setup.py install --prefix=%s" % self.env_path)

        if (code != 0):
            logging.info(out)
            raise RuntimeError("Failed to install %s to %s:" % (package, prefix), code, err)

    def deactivate(self, keep_env):
        """
        Deactivate and optionally remove a conda env at the end of a benchmarking run.
        """
        if not keep_env:
            conda_delete = "conda env remove -q -y --name " + name
            code, out, err = execute_cmd(conda_delete)
            return code


class Slack(object):
    """
    this class encapsulates the logic required to post messages and files to Slack
    """
    def __init__(self, cfg, ca):
        self.cfg = cfg
        self.ca = ca

    def post_message(self, message):
        """
        post a simple message
        """
        payload = {
            "attachments": [
                {
                    "fallback": message,
                    "pretext":  message,
                    "mrkdwn_in": ["pretext", "fields"]
                }
            ],
            "unfurl_links": "false",
            "unfurl_media": "false"
        }

        return self.post_payload(payload)

    def post_image(self, title, image_url):
        """
        post an image
        """
        payload = {
            "attachments": [
                {
                    "fallback":  title,
                    "pretext":   title,
                    "image_url": image_url
                }
            ],
            "unfurl_links": "true",
            "unfurl_media": "true"
        }

        return self.post_payload(payload)

    def post_payload(self, payload):
        """
        post a payload
        """
        p = json.dumps(payload)
        u = self.cfg["message"]
        if self.ca:
            c = "--cacert %s --capath %s " % (self.ca["cacert"], self.ca["capath"])
        else:
            c = ""

        cmd = "curl -s -X POST -H 'Content-type: application/json' --data '%s' %s %s" % (p, u, c)

        code, out, err = execute_cmd(cmd)
        if code:
            logging.warning("Could not post msg to slack: %d\n%s\n%s", code, out, err)

        return code

    def post_file(self, filename, title):
        """
        post a file to slack
        """
        cmd = "curl -s "

        cmd += "-F file=@%s -F title=%s -F filename=%s -F channels=%s -F token=%s " % \
               (filename, title, filename, self.cfg["channel"], self.cfg["token"])

        if self.ca:
            cmd += "--cacert %s --capath %s " % (self.ca["cacert"], self.ca["capath"])

        cmd += "https://slack.com/api/files.upload"

        code, out, err = execute_cmd(cmd)

        if code:
            logging.warning("Could not post file to slack:\n%s\n%s", out, err)

        return code


class BenchmarkDatabase(object):
    """
    this class encapsulates logic that operates on a benchmark database
    """
    def __init__(self, name):
        self.name = name
        self.dbname = name+".db"
        self.connection = sqlite3.connect(self.dbname)

        logging.info('Connected to database: ' + os.path.abspath(self.dbname))

    def _ensure_commits(self):
        """
        if the commit tables have not been created yet, create them
        """
        with self.connection as c:
            # a table containing the last benchmarked commit for each trigger
            # repository
            c.execute("CREATE TABLE if not exists LastCommits"
                      " (Trigger TEXT UNIQUE, LastCommitID TEXT)")

            # a table containing the commit ID for each trigger repository
            # for a given benchmark run (specified by DateTime)
            c.execute("CREATE TABLE if not exists Commits"
                      " (DateTime INT, Trigger TEXT, CommitID TEXT,"
                      "  PRIMARY KEY (DateTime, Trigger))")

    def _ensure_benchmark_data(self):
        """
        if the benchmark data tables have not been created yet, create them
        """
        with self.connection as c:
            # a table containing the status, elapsed time and memory usage for each
            # benchmark in a run
            c.execute("CREATE TABLE if not exists BenchmarkData"
                      " (DateTime INT, Spec TEXT, Status TEXT, Elapsed REAL, Memory REAL,"
                      "  LoadAvg1m REAL, LoadAvg5m REAL, LoadAvg15m REAL, PRIMARY KEY (DateTime, Spec))")

            # a table containing the versions of all installed dependencies for a run
            c.execute("CREATE TABLE if not exists InstalledDeps"
                      " (DateTime INT,  InstalledDep TEXT, Version TEXT,"
                      "  PRIMARY KEY (DateTime, InstalledDep))")

    def get_last_commit(self, trigger):
        """
        Check the database for the most recent commit that was benchmarked
        for the trigger repository
        """
        self._ensure_commits()

        c = self.connection.cursor()
        c.execute("SELECT LastCommitID FROM LastCommits WHERE Trigger == ?", (trigger,))

        rows = c.fetchall()

        if rows:
            return rows[0][0]
        else:
            return ''

    def update_commits(self, commits, timestamp):
        """
        update commits
        """
        self._ensure_commits()

        with self.connection as c:
            for trigger, commit in commits.items():
                logging.info('INSERTING COMMIT %s %s', trigger, commit)
                c.execute('INSERT OR REPLACE INTO LastCommits VALUES (?, ?)', (trigger, str(commit)))
                c.execute('INSERT INTO Commits VALUES (?, ?, ?)', (timestamp, trigger, str(commit)))

    def add_benchmark_data(self, commits, filename, installed):
        """
        Insert benchmarks results into BenchmarkData table.
        Create the table if it doesn't already exist.
        """
        self._ensure_benchmark_data()

        data_added = False

        with self.connection as c:
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)

                for row in reader:
                    logging.info('INSERTING BenchmarkData %s' % str(row))
                    try:
                        spec = row[1].rsplit('/', 1)[1]  # remove path from benchmark file name
                        c.execute("INSERT INTO BenchmarkData VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                                  (row[0], spec, row[2], float(row[3]), float(row[4]),
                                   float(row[5]), float(row[6]), float(row[7])))
                        data_added = True
                    except IndexError:
                        logging.warning("Invalid benchmark data found in results:\n %s", str(row))

            if data_added:
                timestamp = row[0]  # row[0] is the timestamp for this set of benchmark data
                self.update_commits(commits, timestamp)
                for dep, ver in installed.items():
                    c.execute("INSERT INTO InstalledDeps VALUES(?, ?, ?)", (timestamp, dep, ver))

    def check_benchmarks(self, timestamp=None, threshold=20.):
        """
        Check the benchmark data from the given timestamp for any benchmark with a
        significant change (greater than threshold) in elapsed time or memory usage.
        If no timestamp is given then check the most recent benchmark data.
        """
        self._ensure_benchmark_data()

        logging.info("Checking benchmarks for timestamp: %s", timestamp)

        if timestamp is None:
            with self.connection as c:
                for row in c.execute("SELECT * FROM BenchmarkData "
                                     "ORDER BY DateTime DESC LIMIT 1"):
                    timestamp = row[0]  # row[0] is the timestamp for this set of benchmark data

        if timestamp is None:
            msg = "No benchmark data found"
            logging.warning(msg)
            return [], []

        date_str = datetime.fromtimestamp(timestamp)

        curr_data = self.get_data_for_timestamp(timestamp)
        if not curr_data:
            msg = "No benchmark data found for timestamp %d (%s)" % (timestamp, date_str)
            logging.warning(msg)
            return [], []

        prev_time = None
        with self.connection as c:
            for row in c.execute("SELECT * FROM BenchmarkData "
                                 "WHERE DateTime<? and Status=='OK' "
                                 "ORDER BY DateTime DESC LIMIT 1", (timestamp,)):
                prev_time = row[0]  # row[0] is the timestamp for this set of benchmark data

        if not prev_time:
            msg = "No benchmark data found previous to timestamp %d (%s)" % (timestamp, date_str)
            logging.warning(msg)
            return [], []

        prev_data = self.get_data_for_timestamp(prev_time)

        cpu_messages = []
        mem_messages = []

        for i in range(len(curr_data["spec"])):
            curr_spec    = curr_data["spec"][i]

            curr_elapsed = curr_data["elapsed"][i]
            curr_memory  = curr_data["memory"][i]
            curr_load1   = curr_data["load_1m"][i]
            curr_load5   = curr_data["load_5m"][i]
            curr_load15  = curr_data["load_15m"][i]

            if len(prev_data["elapsed"]) > i:
                prev_elapsed = prev_data["elapsed"][i]
                prev_memory  = prev_data["memory"][i]
                prev_load1   = prev_data["load_1m"][i]
                prev_load5   = prev_data["load_5m"][i]
                prev_load15  = prev_data["load_15m"][i]

                time_delta   = curr_elapsed - prev_elapsed
                mem_delta    = curr_memory  - prev_memory

                if "url" in conf:
                    link = "<%s|%s>" % (conf["url"]+self.name+'/'+curr_spec, curr_spec.split(".")[-1])
                else:
                    link = curr_spec.split(".")[-1]

                pct_change = 100.*time_delta/prev_elapsed
                if abs(pct_change) >= threshold:
                    inc_or_dec = "decreased" if (pct_change < 0) else "increased"
                    msg = "%s %s by %4.1f%%: %5.2f (load avg = %3.1f, %3.1f, %3.1f) vs. %5.2f (load avg = %3.1f, %3.1f, %3.1f)" \
                        % (link, inc_or_dec, abs(pct_change),
                           curr_elapsed, curr_load1, curr_load5, curr_load15,
                           prev_elapsed, prev_load1, prev_load5, prev_load15)
                    cpu_messages.append(msg)

                pct_change = 100.*mem_delta/prev_memory
                if abs(pct_change) >= threshold:
                    inc_or_dec = "decreased" if (pct_change < 0) else "increased"
                    msg = "%s %s by %4.1f%% (%5.2f  vs. %5.2f)" \
                        % (link, inc_or_dec, abs(pct_change), curr_memory, prev_memory)
                    mem_messages.append(msg)

        return cpu_messages, mem_messages

    def get_data_for_timestamp(self, timestamp):
        """
        get benchmark data for the given timestamp
        """
        data = {}

        with self.connection as c:
            for row in c.execute("SELECT * FROM BenchmarkData "
                                 "WHERE DateTime=? and Status=='OK' "
                                 "ORDER BY DateTime", (timestamp,)):
                data.setdefault('timestamp', []).append(row[0])
                data.setdefault('spec', []).append(row[1])
                data.setdefault('status', []).append(row[2])
                data.setdefault('elapsed', []).append(row[3])
                data.setdefault('memory', []).append(row[4])
                data.setdefault('load_1m', []).append(row[5])
                data.setdefault('load_5m', []).append(row[6])
                data.setdefault('load_15m', []).append(row[7])

        return data

    def get_data_for_spec(self, spec, since=None):
        """
        get benchmark data for the given spec
        """
        data = {}

        c = self.connection.cursor()

        if since:
            c.execute("SELECT * FROM BenchmarkData "
                      "WHERE Spec=? and Status=='OK' and DateTime>? "
                      "ORDER BY DateTime", (spec, since))
        else:
            c.execute("SELECT * FROM BenchmarkData "
                      "WHERE Spec=? and Status=='OK' "
                      "ORDER BY DateTime", (spec,))

        rows = c.fetchall()

        for row in rows:
            data.setdefault('timestamp', []).append(row[0])
            data.setdefault('status', []).append(row[2])
            data.setdefault('elapsed', []).append(row[3])
            data.setdefault('memory', []).append(row[4])
            data.setdefault('LoadAvg1m', []).append(row[5])
            data.setdefault('LoadAvg5m', []).append(row[6])
            data.setdefault('LoadAvg15m', []).append(row[7])

        return data

    def dump(self):
        """
        dump database to SQL file
        """
        with open(self.dbname+'.sql', 'w') as f:
            for line in self.connection.iterdump():
                f.write('%s\n' % line)

    def plot_all(self, show=False, save=True):
        """
        generate a history plot of each benchmark
        """
        self._ensure_benchmark_data()

        specs = []
        with self.connection as c:
            for row in c.execute("SELECT DISTINCT Spec FROM BenchmarkData"):
                specs.append(row[0])

        filenames = []
        for spec in specs:
            filenames.append(self.plot_benchmark_data(spec, show=show, save=save))

        return [f for f in filenames if f is not None]

    def get_specs(self):
        specs = []
        with self.connection as c:
            for row in c.execute("SELECT DISTINCT Spec FROM BenchmarkData"):
                specs.append(row[0])
        return specs

    def plot_benchmark_data(self, spec=None, show=False, save=False):
        """
        generate a history plot for a benchmark
        """
        import numpy as np

        logging.info('plot: %s', spec)

        self._ensure_benchmark_data()

        filename = None

        try:
            if not show:
                import matplotlib
                matplotlib.use('Agg')
            from matplotlib import pyplot, ticker

            data = self.get_data_for_spec(spec)

            if not data:
                logging.warning("No data to plot for %s", spec)
                return

            timestamp = np.array(data['timestamp'])
            elapsed   = np.array(data['elapsed'])
            maxrss    = np.array(data['memory'])

            fig, a1 = pyplot.subplots()
            a1.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
            x = np.array(range(len(timestamp)))

            a1.plot(x, elapsed, 'b-')
            a1.set_xlabel('run#')
            a1.set_ylabel('elapsed', color='b')
            a1.set_ylim(0, max(elapsed)*1.15)
            for tl in a1.get_yticklabels():
                tl.set_color('b')

            a2 = a1.twinx()
            a2.plot(x, maxrss, 'r-')
            a2.set_ylabel('maxrss', color='r')
            a2.set_ylim(0, max(maxrss)*1.15)
            for tl in a2.get_yticklabels():
                tl.set_color('r')

            label = spec.rsplit(':', 1)[1]
            pyplot.title(label.replace(".benchmark_", ": "))

            if show:
                pyplot.show()

            if save:
                filename = spec.replace(":", "_") + ".png"
                pyplot.savefig(filename)
                code, out, err = execute_cmd("chmod 644 " + filename)

        except ImportError:
            logging.info("numpy and matplotlib are required to plot benchmark data.")
        except err:
            raise err

        return filename

    def plot_benchmarks(self, show=False, save=False):
        """
        generate a history plot for this projects benchmarks
        """
        import numpy as np

        logging.info('plot: %s', self.name)
        filenames = []

        try:
            if not show:
                import matplotlib
                matplotlib.use('Agg')
            from matplotlib import pyplot, dates

            color_map = pyplot.get_cmap('rainbow')

            mondays = dates.WeekdayLocator(dates.MONDAY)    # major ticks on the mondays
            weekFmt = dates.DateFormatter('%b %d')          # e.g., Jan 12

            self._ensure_benchmark_data()

            # select only the specs that have more than one data point
            specs = self.get_specs()

            select_specs = []
            for spec in specs:
                data = self.get_data_for_spec(spec)
                if data and len(data['elapsed']) > 1:
                    select_specs.append(spec)

            specs = select_specs

            specs_per_plot = 9

            plot_count = int(math.ceil(len(specs)/specs_per_plot))

            for plot_no in range(plot_count):
                pyplot.figure()

                # select up to 'specs_per_plot' specs to plot
                plot_specs = specs[:specs_per_plot]
                specs = specs[specs_per_plot:]

                # initialize max values to normalize data
                max_elapsed = 0
                max_memory = 0

                num_specs = len(plot_specs)
                color_cycle = iter([color_map(1.*i/num_specs) for i in range(num_specs)])

                for spec in plot_specs:
                    # get benchmark data for the last 6 weeks
                    since = time.time() - 6*7*24*60*60
                    data = self.get_data_for_spec(spec, since=since)

                    if data:
                        datetimes = [datetime.fromtimestamp(t) for t in data['timestamp']]

                        timestamp = dates.date2num(datetimes)
                        elapsed   = np.array(data['elapsed'])
                        memory    = np.array(data['memory'])

                        max_elapsed = max(max_elapsed, np.max(elapsed))
                        max_memory  = max(max_memory, np.max(memory))

                        color = next(color_cycle)

                        a1 = pyplot.subplot(3, 1, 1)
                        pyplot.plot_date(timestamp, elapsed/max_elapsed, '.-', color=color, label=spec)
                        pyplot.ylabel('elapsed time')

                        a2 = pyplot.subplot(3, 1, 2)
                        pyplot.plot_date(timestamp, memory/max_memory, '.-', color=color, label=spec)
                        pyplot.ylabel('memory usage')

                        # format the ticks
                        a1.set_xticks([])
                        a2.xaxis.set_minor_locator(mondays)
                        a2.xaxis.set_major_formatter(weekFmt)
                        for tick in a2.xaxis.get_major_ticks():
                            tick.label.set_fontsize('x-small')
                        # pyplot.xticks(rotation=45)

                        a1.set_ylim(-0.1, 1.1)
                        a2.set_ylim(-0.1, 1.1)

                pyplot.legend(plot_specs, loc=9, prop={'size': 8}, bbox_to_anchor=(0.5, -0.2))

                if show:
                    pyplot.show()

                if save:
                    # unique filename for every hour, recycled every day (24hr cache on Slack?)
                    from time import localtime, strftime
                    filename = self.name + strftime("_%H_", localtime()) + str(plot_no) + ".png"
                    pyplot.savefig(filename)
                    code, out, err = execute_cmd("chmod 644 " + filename)
                    filenames.append(filename)

        except ImportError:
            logging.info("numpy and matplotlib are required to plot benchmark data.")

        return filenames

    def backup(self):
        """
        create a local backup database, rsync it to destination
        """
        name = self.dbname

        # first save the previous backup, if any
        save_cmd = "mv -f " + name + ".bak " + name + ".prev"
        code, out, err = execute_cmd(save_cmd)

        # create a new backup of the current database and upload it
        backup_cmd = "sqlite3 " + name + " \".backup " + name + ".bak\""
        code, out, err = execute_cmd(backup_cmd)
        if not code:
            try:
                dest = conf["data"]["upload"]
                rsync_cmd = "rsync -zvh " + name + ".bak " + dest + "/" + name
                code, out, err = execute_cmd(rsync_cmd)
            except KeyError:
                pass  # remote backup not configured
            except:
                logging.error("ERROR attempting remote backup")
                logging.error(traceback.format_exc())


class BenchmarkRunner(object):
    """
    this class encapsulates the logic required to conditionally run
    a set of benchmarks if a trigger repository is updated
    """
    def __init__(self, project):
        self.project = project

        # load the database
        self.db = BenchmarkDatabase(project["name"])

        if "slack" in conf:
            self.slack = Slack(conf["slack"], conf.get("ca"))
        else:
            self.slack = None

    def run(self, force=False, keep_env=False, unit_tests=False):
        """
        run benchmarks if project or any of its dependencies have changed
        """
        logging.info("\n************************************************"
                     "\nRunning benchmarks for %s"
                     "\n************************************************"
                     % self.project["name"])

        project = self.project
        db = self.db

        current_commits = {}
        triggered_by = []

        # initialize log file
        timestr = time.strftime("%Y%m%d-%H%M%S")
        run_name = project["name"] + "_" + timestr
        init_log_file(run_name)

        # remove any previous repo_dir for this project so we start fresh
        remove_dir(conf["repo_dir"])

        # determine if a new benchmark run is needed, this may be due to the
        # project repo or a trigger repo being updated or the `force` option
        if force:
            triggered_by.append('force')

        triggers = project.get("triggers", [])

        for trigger in triggers + [project["repository"]]:
            trigger = os.path.expanduser(trigger)
            # for the project repository, we may want a particular branch
            if trigger is project["repository"]:
                branch = project.get("branch", None)
            else:
                branch = None
            # check each trigger for any update since last run
            with repo(trigger, branch):
                msg = 'checking trigger ' + trigger + ' ' + branch if branch else ''
                logging.info(msg)
                current_commits[trigger] = get_current_commit(trigger)
                logging.info("Curr CommitID: %s", current_commits[trigger])
                last_commit = str(db.get_last_commit(trigger))
                logging.info("Last CommitID: %s", last_commit)
                if (last_commit != current_commits[trigger]):
                    logging.info("There has been an update to %s\n", trigger)
                    triggered_by.append(trigger)

        # if new benchmark run is needed:
        # - create and activate a clean env
        # - run unit tests, if desired
        # - run the benchmark
        # - save benchmark results to database
        # - clean up env and repos
        # - back up database
        if triggered_by:
            logging.info("Benchmark triggered by updates to: %s", str(triggered_by))
            trigger_msg = self.get_trigger_message(triggered_by, current_commits)

            dependencies = project.get("dependencies", [])

            # if unit testing fails, the current set of commits will be recorded in fail_file
            fail_file = os.path.join(conf['working_dir'], project["name"]+".fail")

            # start out assuming we have a good set of commits
            good_commits = True

            # if unit testing is enabled, then check that we have not already failed unit testing
            if unit_tests:
                if os.path.exists(fail_file):
                    good_commits = False
                    failed_commits = read_json(fail_file)
                    for key in current_commits:
                        if current_commits[key] != failed_commits[key]:
                            # there has been a new commit, set flag to run and delete fail_file
                            logging.info("found new commit for %s", key)
                            logging.info("old commit: %s", failed_commits[key])
                            logging.info("new commit %s:", current_commits[key])
                            good_commits = True
                            os.remove(fail_file)
                            break

                if not good_commits:
                    logging.info("This set of commits has already failed unit testing.")

            if good_commits or 'force' in triggered_by:

                # activate conda env
                conda_env = CondaEnv(run_name, dependencies, triggers)

                with repo(project["repository"], project.get("branch", None)):
                    logging.info("========== INSTALL PROJ & RUN ==========")

                    # install project, with any specified extras
                    extras = project.get("extras", "")
                    conda_env.install("."+extras, options="-e")

                    # run the unit tests if requested and record current_commits if it fails
                    if unit_tests:
                        with conda(conda_env):
                            rc = self.run_unittests(run_name, trigger_msg)
                        if rc:
                            write_json(fail_file, current_commits)
                            good_commits = False

                    # if we still show good commits, run benchmarks and add data to database
                    if good_commits:

                        # get list of installed dependencies
                        installed_deps = {}
                        with conda(conda_env):
                            rc, out, err = execute_cmd("conda list")
                        for line in out.split('\n'):
                            name_ver = line.split(" ", 1)
                            if len(name_ver) == 2:
                                installed_deps[name_ver[0]] = name_ver[1]

                        csv_file = run_name+".csv"
                        with conda(conda_env):
                            rc = self.run_benchmarks(run_name, trigger_msg, csv_file)
                        if rc:
                            write_json(fail_file, current_commits)
                            good_commits = False
                        else:
                            db.add_benchmark_data(current_commits, csv_file, installed_deps)
                            self.post_results(trigger_msg)
                            if conf["remove_csv"]:
                                os.remove(csv_file)

                if good_commits:
                    # if benchmarks didn't fail but there are no commits in database, then
                    # no benchmarks are defined so don't run again for this set of commits
                    if not db.get_last_commit(project["repository"]):
                        write_json(fail_file, current_commits)
                    else:
                        # back up and transfer database
                        db.backup()

                # clean up environment
                conda_env.deactivate(keep_env)

        # close the log file for this run
        close_log_file()

    def post_results(self, trigger_msg):
        """
        generate plots and post a message to slack detailing benchmark results
        """
        db = self.db
        name = self.project["name"]

        # generate summary plots if requested and upload if image location is provided
        image_url = None
        summary_plots = []

        if conf["plot_history"]:
            summary_plots = db.plot_benchmarks(save=True, show=False)
            if conf.get("images") and summary_plots:
                rc = upload(summary_plots, conf["images"]["upload"])
                if rc == 0:
                    image_url = conf["images"]["url"]

        # if slack info is provided, post message and plots to slack
        if self.slack:
            # post message that benchmarks were run
            self.slack.post_message(trigger_msg)

            # post summary plots
            if summary_plots:
                for plot_file in summary_plots:
                    self.slack.post_image("", "/".join([image_url, plot_file]))
            else:
                self.slack.post_message("No benchmark history plots generated.")

            # check benchmarks for significant changes & post any resulting messages
            cpu_messages, mem_messages = db.check_benchmarks()
            # notify = "<!channel> The following %s benchmarks had a significant change in %s:\n"
            notify = "The following %s benchmarks had a significant change in %s:\n"

            # post max_messages at a time
            max_messages = 9

            if cpu_messages:
                self.slack.post_message(notify % (name, "elapsed time"))
                while cpu_messages:
                    msg = '\n'.join(cpu_messages[:max_messages])
                    self.slack.post_message(msg)
                    cpu_messages = cpu_messages[max_messages:]
                if os.path.exists("top.txt"):
                    self.slack.post_file("top.txt", "Top 10 CPU before and after")

            if mem_messages:
                self.slack.post_message(notify % (name, "memory usage"))
                while mem_messages:
                    msg = '\n'.join(mem_messages[:max_messages])
                    self.slack.post_message(msg)
                    mem_messages = mem_messages[max_messages:]

    def run_unittests(self, run_name, trigger_msg):
        """
        Use testflo to run unit tests
        """
        testflo_cmd = "testflo -n 1 --pre_announce"

        # run testflo command
        code, out, err = execute_cmd(testflo_cmd)
        logging.info(out)
        logging.warning(err)

        if code:
            # an expected failure will return an error code, check fail count
            for line in open("testflo_report.out"):
                if line.startswith("Failed:"):
                    if line.split()[1] == "0":
                        print("testflo returned non-zero but there were no failures.")
                        code = 0
                    break

        # if failure, post to slack
        if code and self.slack:
            self.slack.post_message(trigger_msg + "However, unit tests failed... <!channel>")
            fail_msg = "\"%s : regression testing has failed. See attached results file.\"" % self.project["name"]
            self.slack.post_file("testflo_report.out", fail_msg)

        return code

    def run_benchmarks(self, run_name, trigger_msg, csv_file):
        """
        Use testflo to run benchmarks
        """
        benchmark_cmd = conf.get("benchmark_cmd")
        if benchmark_cmd:
            benchmark_cmd = "%s %s %s" % (benchmark_cmd, run_name, csv_file)
        else:
            benchmark_cmd = "testflo -n 1 -bv -d %s" % csv_file

        code, out, err = execute_cmd(benchmark_cmd)

        # if failure, post to slack
        if code and self.slack:
            self.slack.post_message(trigger_msg + "However, benchmarks failed... <!channel>")
            fail_msg = "\"%s : benchmarking has failed. See attached results file.\"" %  self.project["name"]
            self.slack.post_file("testflo_report.out", fail_msg)

        return code

    def get_trigger_message(self, triggered_by, current_commits):
        """
        list specific commits (in link form) that triggered benchmarks to run
        """
        name = self.project["name"]

        if "url" in conf:
            pretext = "<%s|%s> benchmarks triggered by " % (conf["url"]+name, name)
        else:
            pretext = "*%s* benchmarks triggered by " % name

        if "force" in triggered_by:
            pretext = pretext + "force:\n"
        else:
            links = []
            # add the specific commit information to each trigger
            for url in triggered_by:
                if "bitbucket" in url:
                    commit = "/commits/"
                else:
                    commit = "/commit/"
                links.append(url + commit + str(current_commits[url]).strip('\n'))

            # insert proper formatting so long URL text is replaced by short trigger-name hyperlink
            links = ["<%s|%s>" % (url.replace("git@github.com:", "https://github.com/"), url.split('/')[-3])
                     for url in links]

            pretext = pretext + "updates to: " + ", ".join(links) + "\n"

        return pretext


#
# command line
#

def _get_parser():
    """
    Returns a parser to handle command line args.
    """

    parser = ArgumentParser()
    parser.usage = "benchmark [options]"

    parser.add_argument('projects', metavar='project', nargs='*',
                        help='project to benchmark (references a JSON file in the working directory)')

    parser.add_argument('-f', '--force', action='store_true', dest='force',
                        help='do the benchmark even if nothing has changed')

    parser.add_argument('-u', '--unit-tests', action='store_true', dest='unit_tests',
                        help='run the unit tests before running the benchmarks.')

    parser.add_argument('-k', '--keep-env', action='store_true', dest='keep_env',
                        help='keep the created conda env after execution (usually for troubleshooting purposes)')

    parser.add_argument('-p', '--plot', metavar='SPEC', action='store', dest='plot',
                        help='plot benchmark history for SPEC')

    parser.add_argument('-c', '--check', action='store_true', dest='check',
                        help='check the most recent benchmark data for significant change')

    parser.add_argument('-d', '--dump', action='store_true', dest='dump',
                        help='dump the contents of the database to an SQL file')

    return parser


def main(args=None):
    """
    process command line arguments and perform requested task
    """
    if args is None:
        args = sys.argv[1:]

    options = _get_parser().parse_args(args)

    # read local configuration if available
    try:
        conf.update(read_json("benchmark.cfg"))
    except IOError:
        pass

    # initalize logging to stdout
    init_logging()

    # perform the requested operation for each project
    with cd(conf["working_dir"]):
        for project in options.projects:

            # get project info
            if project.endswith(".json"):
                project_file = project
            else:
                project_file = project+".json"
            project_info = read_json(project_file)

            if project_info.get("skip") and not options.force:
                continue

            project_name = os.path.basename(project_file).rsplit('.', 1)[0]
            project_info["name"] = project_name

            init_env(project_info)

            # perform requested action, or just run benchmarks
            if options.plot:
                db = BenchmarkDatabase(project_name)
                if options.plot == 'all':
                    db.plot_benchmarks(save=True)
                else:
                    db.plot_benchmark_data(options.plot, show=True)
            elif options.dump:
                db = BenchmarkDatabase(project_name)
                db.dump()
            elif options.check:
                db = BenchmarkDatabase(project_name)
                messages = db.check_benchmarks()
                for msg in messages:
                    logging.info(msg)
            else:
                # use a different repo directory for each project
                conf["repo_dir"] = os.path.expanduser(
                    os.path.join(conf["working_dir"], (project_name+"_repos")))

                bm = BenchmarkRunner(project_info)
                bm.run(options.force, options.keep_env, options.unit_tests)


if __name__ == '__main__':
    sys.exit(main())
