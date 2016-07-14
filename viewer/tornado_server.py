#!/usr/bin/env python
import os
from datetime import datetime

import tornado.ioloop
import tornado.web
from benchmark import BenchmarkDatabase


#database_dir = os.path.abspath(os.path.dirname(__file__))
database_dir = "/home/openmdao/webapps/benchmark_data_server/"


class ProjectHandler(tornado.web.RequestHandler):
    def get(self, project):
        """
        Display benchmarks for project.
        If no project is specified, show list of projects.
        """
        dbs = [f for f in os.listdir(database_dir) if f.endswith(".db")]

        if not project:
            dbs = [f.rsplit(".")[0] for f in dbs]
            self.render("main_template.html", dbs=dbs)
        elif (project+".db") not in dbs:
            self.finish("<html><body>%s is not a valid project</body></html>" % project)
        else:
            db = BenchmarkDatabase(os.path.join(database_dir, project))
            specs = db.get_specs()
            dates = []
            for spec in specs:
                for row in db.cursor.execute('SELECT DateTime FROM BenchmarkData WHERE Spec==? ORDER BY DateTime DESC LIMIT 1', (spec,)):
                    dates.append(row[0])

            def date(timestamp):
                return str(datetime.fromtimestamp(timestamp))

            self.render("proj_template.html", title=project, spec=specs, date=date, dates=dates)


class SpecHandler(tornado.web.RequestHandler):
    def get(self, project, spec):
        """
        Display history for specific benchmark.
        """
        db = BenchmarkDatabase(os.path.join(database_dir, project))

        data = {}
        for row in db.cursor.execute('SELECT * FROM BenchmarkData WHERE Spec=? and Status=="OK" ORDER BY DateTime', (spec,)):
            data.setdefault("timestamp", []).append(row[0])
            data.setdefault("status", []).append(row[2])
            data.setdefault("elapsed", []).append(row[3])
            data.setdefault("memory", []).append(row[4])
            data.setdefault("LoadAvg1m", []).append(row[5])
            data.setdefault("LoadAvg5m", []).append(row[6])
            data.setdefault("LoadAvg15m", []).append(row[7])

        commits = {}
        for timestamp in data["timestamp"]:
            tmp_list = []
            for row in db.cursor.execute('SELECT * FROM Commits WHERE DateTime==? ORDER BY DateTime', (timestamp,)):
                # row[0] is timestamp, row[1] is trigger, row[2] is commit
                prefix = ""
                name = row[1].rsplit('/', 1)[1]
                if "github" in row[1]:
                    prefix = row[1].strip().replace(':', '/').replace('git@', 'http://')
                if "bitbucket" in row[1]:
                    prefix = row[1]
                commit = row[2]
                tmp_list.append((name, commit, prefix))
            commits[timestamp] = tmp_list

        data["commits"] = commits
        data["datestr"] = ["{:%Y-%m-%d %H:%M}".format(datetime.fromtimestamp(t)) for t in data["timestamp"]]

        if not data:
            print("No data to plot for %s" % spec)
            return "No data for %s %s " % (project, spec)
        else:
            bench_title = "Benchmark Results for " + spec
            self.render("spec_template.html", title=bench_title, items=data)


if __name__ == "__main__":
    app = tornado.web.Application([
        (r'/(favicon.ico)', tornado.web.StaticFileHandler, {"path": ""}),
        (r"/(.*)/(.*)",     SpecHandler),
        (r"/(.*)",          ProjectHandler),
    ], debug=True)

    app.listen(18309)
    tornado.ioloop.IOLoop.current().start()