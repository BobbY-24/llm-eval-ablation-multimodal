"""Registry of local readers. Optional media/parser dependencies stay lazy."""
from .vpt import VPTAdapter
from .minerl import MineRLAdapter
from .csgo import CSGOAdapter
from .terraria import TerrariaAdapter
from .proact import ProactAdapter
from .starcraftimage import StarCraftImageAdapter
from .slippi import SlippiAdapter
from .dota2vis import Dota2VisAdapter
from .aoe2 import AoE2Adapter

ADAPTERS = {"vpt": VPTAdapter, "minerl": MineRLAdapter, "csgo": CSGOAdapter,
            "terraria": TerrariaAdapter, "proact": ProactAdapter,
            "starcraftimage": StarCraftImageAdapter, "slippi": SlippiAdapter,
            "dota2vis": Dota2VisAdapter, "aoe2": AoE2Adapter}


def create_adapter(name, root):
    try:
        adapter = ADAPTERS[name]
    except KeyError as error:
        raise ValueError(f"Unknown dataset adapter: {name}. Choose from {', '.join(ADAPTERS)}") from error
    return adapter(root)
