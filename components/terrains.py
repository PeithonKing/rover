"""
components/terrains.py
======================
Pluggable terrain worlds for 6-wheel rover simulation.

Defines:
- BaseTerrain: Abstract base class specifying the xml_path property.
- FlatTerrain: Flat horizontal terrain with infinite ground plane,
  skybox, lighting, and target waypoint marker site.
  Loads assets/worlds/world_flat.xml with fallback to 3D_files/mujoco/scene.xml.

All comments and strings strictly use 7-bit ASCII characters.
"""

from abc import ABC, abstractmethod
import os
from typing import Any, Optional


class BaseTerrain(ABC):
    """
    Abstract base class for environment terrain worlds.
    """

    @property
    @abstractmethod
    def xml_path(self) -> str:
        """Return the absolute filesystem path to the MuJoCo XML world file."""
        pass

    def randomize(self, env: Any) -> None:
        """
        Optional terrain randomization callback during env.reset().
        """
        pass


class FlatTerrain(BaseTerrain):
    """
    Flat terrain world definition.
    Loads assets/worlds/world_flat.xml with fallback to 3D_files/mujoco/scene.xml.
    """

    def __init__(self, xml_path: Optional[str] = None) -> None:
        if xml_path is not None:
            self._xml_path = os.path.abspath(xml_path)
        else:
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = (
                os.path.dirname(cur_dir)
                if os.path.basename(cur_dir) == "components"
                else cur_dir
            )
            while base_dir and not (
                os.path.exists(os.path.join(base_dir, "assets"))
                or os.path.exists(os.path.join(base_dir, "3D_files"))
            ):
                parent = os.path.dirname(base_dir)
                if parent == base_dir:
                    break
                base_dir = parent

            world_flat = os.path.join(base_dir, "assets", "worlds", "world_flat.xml")
            fallback = os.path.join(base_dir, "3D_files", "mujoco", "scene.xml")
            if os.path.exists(world_flat):
                self._xml_path = os.path.abspath(world_flat)
            elif os.path.exists(fallback):
                self._xml_path = os.path.abspath(fallback)
            else:
                raise FileNotFoundError(
                    f"Neither world_flat.xml ({world_flat}) nor fallback scene.xml ({fallback}) exists."
                )

    @property
    def xml_path(self) -> str:
        """Absolute path to world XML file."""
        return self._xml_path

    def randomize(self, env: Any) -> None:
        """Flat terrain does not require runtime randomization."""
        pass
