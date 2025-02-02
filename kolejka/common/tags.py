# vim:ts=4:sts=4:sw=4:expandtab

import os
import re
import sys
from collections import Counter

from kolejka.common.gpu import gpu_stats

def parse_cpu_name(name):
    name = re.sub(r'@.*', '', name.lower())
    name = re.sub(r'[(][^()]*[)]', '', name)
    name = [ n for n in re.split(r'\s+', name) if n ]
    name = [ n for n in name if not re.match('[0-9.+-]*ghz', n) ]
    name = [ n for n in name if not n == 'mobile' ]
    name = [ n for n in name if not n == 'extreme' ]
    name = [ n for n in name if not n == '-' ]
    name = [ n for n in name if not n == 'cpu' ]
    name = [ n for n in name if not n == 'apu' ]
    name = [ n for n in name if not n == 'gen' ]
    name = [ n for n in name if not n == 'solo' ]
    name = [ n for n in name if not n == 'dual' ]
    name = [ n for n in name if not n == 'duo' ]
    name = [ n for n in name if not n == 'trile' ]
    name = [ n for n in name if not n == 'quad' ]
    name = [ n for n in name if not n == 'eight' ]
    name = [ n for n in name if not re.match('intel.*', n) ]
    name = [ n for n in name if not re.match('amd.*', n) ]
    name = [ n for n in name if not re.match('core.*', n) ]
    name = [ n for n in name if not re.match('dual-?core.*', n) ]
    name = [ n for n in name if not re.match('triple-?core.*', n) ]
    name = [ n for n in name if not re.match('quad-?core.*', n) ]
    name = [ n for n in name if not re.match('eight-?core.*', n) ]
    name = ' '.join(name)
    return name

def cpu_tags():
    count = 0
    vendor = None
    name = None
    tags = set()
    vendors = {
        'GenuineIntel' : 'intel',
        'AuthenticAMD' : 'amd',
    }
    with open('/proc/cpuinfo') as cpuinfo_file:
        for line in cpuinfo_file.readlines():
            line = re.sub(r'\s*:\s*', ':', line)
            line = re.sub(r'\s+', ' ', line)
            line = line.strip()
            line = line.split(':')
            if len(line) >= 2:
                key = line[0]
                val = ':'.join(line[1:])
                if key.lower() == 'processor':
                    count += 1
                if key.lower() == 'vendor_id' and val in vendors:
                    vendor = vendors[val]
                    tags.add('cpu:'+vendor)
                if key.lower() == 'model name':
                    name = parse_cpu_name(val)
                    tags.add('cpu:'+name)
    if count:
        tags.add('cpus:'+str(count))
    return tags

def gpu_tags():
    tags = set()
    counts_per_model = Counter()
    stats = gpu_stats().dump().get('gpus', {}).items()

    if len(stats) > 0:
        tags.add('gpu:nvidia')

    for single_count in range(1, len(stats) + 1):
        tags.add(f'gpus:{single_count}')

    for gpu_id, gpu_params in stats:
        counts_per_model[gpu_params.get('id')] += 1

    for model_name, count in counts_per_model.items():
        tags.add(f'gpu:{model_name}')

        for single_count in range(1, count + 1):
            tags.add(f'gpu:{model_name}:{single_count}')

    return tags

def foreman_auto_tags():
    tags = set()
    tags.update(cpu_tags())
    tags.update(gpu_tags())
    return list(tags)
