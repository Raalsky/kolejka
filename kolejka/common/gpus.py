import gpustat

from kolejka.common.limits import KolejkaStats


def gpu_stats():
    gpu_stat = gpustat.GPUStatCollection.new_query()

    stats = KolejkaStats()
    stats.load({
        'gpus': {
            f'{index}': {
                'name': gpu.name,
                'total_memory': f'{gpu.memory_total * 1024 * 1024}b',
                'free_memory': f'{gpu.memory_free * 1024 * 1024}b',
                'available_memory': f'{gpu.memory_available * 1024 * 1024}b'
            } for index, gpu in enumerate(gpu_stat.gpus)
        }
    })

    return stats
#
# with open('metrics.csv', newline='') as csvfile:
# ...     data = csv.DictReader(csvfile)
# ...     for row in data:
# ...             print(row['Metric Name'])

# nv-nsight-cu-cli --import profile.ncu-rep --csv > metrics.csv