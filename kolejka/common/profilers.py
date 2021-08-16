import io
import csv
import shutil
import subprocess

from kolejka.common.parse import parse_int, json_dict_load, json_list_load
from kolejka.common.limits import KolejkaStats

class KolejkaProfilers:
    class GpuProfiler:
        def __init__(self, **kwargs):
            self.load(kwargs)

        def load(self, data, **kwargs):
            args = json_dict_load(data)
            args.update(kwargs)
            self.metrics = json_list_load(args.get('metrics', ["gpu__time_duration.sum"]))
            self.repeat = parse_int(args.get("repeat", 1))

        def dump(self):
            res = dict()
            if self.repeat > 0:
                res['repeat'] = self.repeat
            if self.metrics is not None:
                res['metrics'] = self.metrics
            return res

        def to_collect(self):
            return [
                {
                    "glob": "profile.ncu-rep"
                }
            ]

        def profiler(self):
            return [
                "/usr/local/bin/nv-nsight-cu-cli",
                "--metrics", ",".join(self.metrics),
                "-c", f"{self.repeat}",
                "-o", "profile",
            ]

        def get_nsight_cli(self):
            path = shutil.which('nv-nsight-cu-cli')
            assert path is not None, "Profiler could not import results without nv-nsight-cu-cli"
            return path

        def stats(self, files):
            profile_file = files.get("profile.ncu-rep")

            metrics = subprocess.run(
                [
                    self.get_nsight_cli(),
                    "--import", profile_file,
                    "--csv",
                ],
                stdout=subprocess.PIPE
            )

            metrics_output = str(metrics.stdout, 'utf-8').strip()
            metrics_csv = io.StringIO(metrics_output)
            data = csv.DictReader(metrics_csv)

            gpus = {}
            for row in data:
                device, name, value = row['ID'], row['Metric Name'], row['Metric Value']
                gpus[device] = gpus.get(device, {'usage': {}})
                gpus[device]['usage'].update({
                    f'{name}': str(value)
                })

            stats = KolejkaStats()
            stats.load({
                'gpus': gpus
            })

            return stats

    def __init__(self, **kwargs):
        self.load(kwargs)

    def load(self, data, **kwargs):
        args = json_dict_load(data)
        args.update(kwargs)
        self.gpu = KolejkaProfilers.GpuProfiler()
        self.gpu.load(args.get('gpu', {}))

    def dump(self):
        res = dict()
        res['gpu'] = self.gpu.dump()
        return res

    def to_collect(self):
        return self.gpu.to_collect()

    def profiler(self):
        return self.gpu.profiler()

    def stats(self, files):
        return self.gpu.stats(files)


if __name__ == '__main__':
    profilers = KolejkaProfilers()
    data = {
        "gpu": {
          "metrics": [
            "gpu__time_duration.sum"
          ],
          "repeat": 1
        }
    }
    files = {
        'profile.ncu-rep': '/home/rjankowski/profile.ncu-rep'
    }
    profilers.load(data)
    print(profilers.dump())
    print(profilers.stats(files).dump())
