"""Check exported costs against the installed Nav2 map loader."""

import shutil
import subprocess

import numpy as np
import pytest

pytest.importorskip('rclpy')
from ament_index_python.packages import get_package_prefix

from offline_map_package.semantic_grid import Geometry
from offline_map_package.semantic_mapper import map_pgm, map_yaml


@pytest.fixture(scope='module')
def nav2_loader(tmp_path_factory):
    if shutil.which('g++') is None:
        pytest.skip('g++ is required for the Nav2 map_io integration check')
    prefix = get_package_prefix('nav2_map_server')
    executable = tmp_path_factory.mktemp('map_loader') / 'load_map'
    source = r'''
    #include <iostream>
    #include "nav2_map_server/map_io.hpp"
    int main(int argc, char ** argv) {
      if (argc != 2) return 2;
      nav_msgs::msg::OccupancyGrid map;
      if (nav2_map_server::loadMapFromYaml(argv[1], map) !=
          nav2_map_server::LOAD_MAP_SUCCESS) return 1;
      std::cout << "LOADED";
      for (auto value : map.data) std::cout << " " << int(value);
      std::cout << std::endl;
      return 0;
    }
    '''
    subprocess.run([
        'g++', '-std=c++14', '-x', 'c++', '-', '-I' + prefix + '/include',
        '-L' + prefix + '/lib', '-Wl,-rpath,' + prefix + '/lib',
        '-lmap_io', '-o', str(executable),
    ], input=source, text=True, check=True, capture_output=True)
    return executable


@pytest.mark.parametrize('mode', ['raw', 'trinary'])
def test_nav2_loads_saved_unknowns_and_costs(nav2_loader, tmp_path, mode):
    values = (np.array([[-1, 0, 30], [60, 90, 100]], dtype=np.int8)
              if mode == 'raw' else
              np.array([[-1, 0, 100], [100, -1, 0]], dtype=np.int8))
    geometry = Geometry(0.05, 3, 2, (1.25, -2.5, 0.4))
    image_path = tmp_path / 'layer.pgm'
    yaml_path = tmp_path / 'layer.yaml'
    image_path.write_bytes(map_pgm(values, mode))
    yaml_path.write_text(map_yaml(image_path.name, geometry, mode))
    result = subprocess.run(
        [str(nav2_loader), str(yaml_path)], check=True,
        capture_output=True, text=True)
    line = next(line for line in result.stdout.splitlines() if line.startswith('LOADED'))
    loaded = [int(value) for value in line.split()[1:]]
    assert loaded == values.ravel().tolist()
