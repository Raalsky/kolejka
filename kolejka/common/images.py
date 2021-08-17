import json
import subprocess
from pathlib import Path

from kolejka.common import parse_memory

def check_docker_image_existance(image: str) -> bool:
    docker_inspect_run = subprocess.run(
        ['docker', 'image', 'inspect', image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    return docker_inspect_run.returncode != 0

def pull_docker_image(image: str):
    subprocess.run(['docker', 'pull', image], check=True)

def get_docker_image_size(image: str) -> int:
    docker_inspect_run = subprocess.run(
        ['docker', 'image', 'inspect', '--format', '{{json .Size}}', image],
        stdout=subprocess.PIPE,
        check=True
    )
    return int(json.loads(str(docker_inspect_run.stdout, 'utf-8')))

def remove_docker_image(image: str):
    subprocess.run(['docker', 'image', 'rm', image])

def list_docker_images():
    ls_output = str(
        subprocess.run(
            ['docker', 'image', 'ls', '--format', '"{{.Repository}}:{{.Tag}} {{.Size}}"'],
            stdout=subprocess.PIPE,
            check=True
        ).stdout,
        'utf-8'
    )

    return dict([
        (a.split()[0], parse_memory(a.split()[1]))
        for a in ls_output.split('\n') if a
    ])

def docker_build_image(dockerfile_path: Path, name: str, tag: str = 'latest'):
    subprocess.run(
        ['docker', 'build', '-t', f'{name}:{tag}', '.'],
        cwd=dockerfile_path,
        stdout=subprocess.PIPE,
        check=True
    )
