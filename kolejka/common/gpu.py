import gpustat

from kolejka.common.limits import KolejkaStats

def normalize_name(name: str) -> str:
    return '-'.join(name.lower().split(' ')[1:])

def gpu_stats():
    query = gpustat.GPUStatCollection.new_query()

    stats = KolejkaStats()
    stats.load({
        'gpus': {
            f'{index}': {
                'name': gpu.name,
                'id': normalize_name(gpu.name),
                'total_memory': f'{gpu.memory_total * 1024 * 1024}b',
                'free_memory': f'{gpu.memory_free * 1024 * 1024}b',
                'available_memory': f'{gpu.memory_available * 1024 * 1024}b'
            } for index, gpu in enumerate(query.gpus)
        }
    })

    return stats
