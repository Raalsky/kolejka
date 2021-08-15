# vim:ts=4:sts=4:sw=4:expandtab

import json

from django.conf import settings
from django.contrib.contenttypes.models import ContentType
from django.db import transaction
from django.db.models import F, Count
from django.http import HttpResponse, JsonResponse, HttpResponseForbidden, HttpResponseNotFound, HttpResponseNotAllowed
import django.utils.timezone
from django.views.decorators.csrf import ensure_csrf_cookie

from kolejka.common import KolejkaLimits
from kolejka.common.parse import parse_time
from kolejka.server.response import OKResponse, FAILResponse
from kolejka.server.task.models import Task, Result

@transaction.atomic
def dequeue(request):
    if not request.user.has_perm('task.process_task'):
        return HttpResponseForbidden()
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    content_type = ContentType.objects.get_for_model(Task)
    tasks = list()
    print('OONLY TESTING')
    params = json.loads(str(request.read(), request.encoding or 'utf-8')) 
    concurency = params.get('concurency', 1)
    limits = KolejkaLimits()
    limits.load(params.get('limits', dict()))
    tags = set(params.get('tags', list()))
    resources = KolejkaLimits()
    resources.update(limits)
    image_usage = dict()

    available_tasks = Task.objects.filter(assignee__isnull=True).order_by('time_create')[0:100]
    print(available_tasks, len(available_tasks))
    for t in available_tasks:
        if len(tasks) > concurency:
            print('Breaked at concurency')
            break
        tt = t.task()
        print(tt.limits.dump())
        if len(tasks) > 0 and tt.exclusive:
            print('Breaked at exclusive')
            continue
        if not set(tt.requires).issubset(tags):
            print('Breaked at tags', tt.requires, tags)
            continue
        if resources.cpus is not None and (tt.limits.cpus is None or tt.limits.cpus > resources.cpus):
            print('Breaked at cpus', resources.cpus, tt.limits.cpus)
            continue
        if resources.gpus is not None and (tt.limits.gpus is None or tt.limits.gpus > resources.gpus):
            print('Breaked at gpus')
            continue
        if resources.memory is not None and (tt.limits.memory is None or tt.limits.memory > resources.memory):
            print('Breaked at memory')
            continue
        if resources.swap is not None and (tt.limits.swap is None or tt.limits.swap > resources.swap):
            print('Breaked at swap')
            continue
        if resources.pids is not None and (tt.limits.pids is None or tt.limits.pids > resources.pids):
            print('Breaked at pids')
            continue
        if resources.storage is not None and (tt.limits.storage is None or tt.limits.storage > resources.storage):
            print('Breaked at storage', resources.storage, tt.limits.storage)
            continue
        if resources.image is not None:
            if tt.limits.image is None:
                print('Breaked at image A')
                continue
            image_usage_add = max(image_usage.get(tt.image, 0), tt.limits.image) - image_usage.get(tt.image, 0)
            if image_usage_add > resources.image:
                print(f"Task need {image_usage_add}, Foreman has {resources.image}")
                print('Breaked at image B')
                continue
        if resources.workspace is not None and (tt.limits.workspace is None or tt.limits.workspace > resources.workspace):
            print('Breaked at workspace')
            continue
        if resources.network is not None and (tt.limits.network is None or tt.limits.network and not resources.network):
            print('Breaked at network')
            continue
        if resources.time is not None and (tt.limits.time is None or tt.limits.time > resources.time):
            print('Breaked at time')
            continue

        tasks.append(tt.dump())
        t.assignee = request.user
        t.time_assign = django.utils.timezone.now() 
        t.save()
        if resources.cpus is not None:
            resources.cpus -= tt.limits.cpus
        if resources.gpus is not None:
            resources.gpus -= tt.limits.gpus
        if resources.memory is not None:
            resources.memory -= tt.limits.memory
        if resources.swap is not None:
            resources.swap -= tt.limits.swap
        if resources.pids is not None:
            resources.pids -= tt.limits.pids
        if resources.storage is not None:
            resources.storage -= tt.limits.storage
        if resources.image is not None:
            resources.image -= image_usage_add
            image_usage[tt.image] = max(image_usage.get(tt.image, 0), tt.limits.image)
        if resources.workspace is not None:
            resources.workspace -= tt.limits.workspace
        if tt.exclusive:
            break

    response = dict()
    response['tasks'] = tasks
    return OKResponse(response)

def stats(request):
    if not request.user.is_authenticated:
        return HttpResponseForbidden()
    if request.method != 'GET':
        return HttpResponseNotAllowed(['GET'])
    response = dict()
    response['task_count'] = Task.objects.count()
    response['task_resolved_count'] = Task.objects.filter(result__isnull=False).count()
    response['task_unresolved_count'] = Task.objects.filter(result__isnull=True).count()
    response['task_unassigned_count'] = Task.objects.filter(assignee__isnull=True).count()
    assignees = dict([ (v['assignee_username'], v['task_count']) for v in Task.objects.filter(assignee__isnull=False, result__isnull=True).annotate(assignee_username=F('assignee__username')).values('assignee_username').annotate(task_count=Count('assignee_username')) ])
    response['task_assigned_count'] = sum(assignees.values())
    response['task_assignees'] = assignees
    return OKResponse(response)
