"""Decode main StarCraftImage PNG containers into native unit bags, not RGB."""
from ..core import Sample, Window, Finding, MissingDependency, stable_id, safe_path, require_dependency
from ._common import LocalAdapter, bounded_paths, relative, csv_rows, MAX_ARRAY_BYTES

MAX_SPATIAL_BYTES = MAX_ARRAY_BYTES


def decode_unit_bags(encoded):
    """Exact packed PNG layout documented by sc2image.dataset, without Torch.

    Storage channels: player2 / neutral / player1. Vertical 64-pixel bands:
    unit values, unit IDs, map state. Horizontal blocks are overlap planes.
    Empty all-zero blocks are omitted independently, matching the upstream reader.
    Unit IDs remain native channel IDs; no guessed unit-name mapping is applied.
    """
    np = require_dependency("numpy")
    image = np.asarray(encoded)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] != 192 or image.shape[1] <= 0 or image.shape[1] % 64:
        raise ValueError("Expected encoded StarCraftImage PNG shape (192, 64*K, 3)")
    if image.dtype.name != "uint8":
        raise ValueError("Packed spatial storage must be uint8")
    # Reserve both the Pillow decode buffer and its NumPy view/copy, all output
    # bands together, and a small reduction scratch allowance before allocation.
    materialized = 2 * image.nbytes + 4096
    if materialized > MAX_SPATIAL_BYTES:
        raise ValueError("Encoded spatial image exceeds aggregate memory budget")
    plans = {}
    for player, channel in (("player_1", 2), ("player_2", 0), ("neutral", 1)):
        for band, key in enumerate(("unit_values", "unit_ids", "map_state")):
            if player == "neutral" and key == "map_state":
                continue
            strip = image[band * 64:(band + 1) * 64, :, channel]
            indices = []
            for index in range(image.shape[1] // 64):
                if np.any(strip[:, index * 64:(index + 1) * 64]):
                    materialized += 64 * 64 + 64  # uint8 plane plus index overhead
                    if materialized > MAX_SPATIAL_BYTES:
                        raise ValueError("Decoded spatial bands exceed aggregate memory budget")
                    indices.append(index)
            plans[f"{player}_{key}"] = (band, channel, indices)
    result = {}
    for key, (band, channel, indices) in plans.items():
        # Fill one allocation directly rather than holding copied blocks and a
        # second stacked copy at the same time.
        output = np.empty((len(indices), 64, 64), dtype=image.dtype)
        for destination, index in enumerate(indices):
            output[destination] = image[band * 64:(band + 1) * 64, index * 64:(index + 1) * 64, channel]
        result[key] = output
    for player in ("player_1", "player_2", "neutral"):
        if result[f"{player}_unit_ids"].shape != result[f"{player}_unit_values"].shape:
            raise ValueError(f"Mismatched {player} native unit ID/value planes")
    return result


class StarCraftImageAdapter(LocalAdapter):
    dataset = "starcraftimage"
    game = "StarCraft II"
    declared_modalities = ["spatial", "state"]

    def iter_samples(self):
        for metadata in bounded_paths(self.root, ("metadata.csv",)):
            if metadata.name != "metadata.csv":
                continue
            folder = metadata.parent
            for row in csv_rows(metadata):
                if not all(k in row for k in ("global_idx", "replay_name", "window_idx")):
                    raise ValueError("StarCraftImage metadata requires global_idx, replay_name, window_idx")
                filename = f'idx_{row["global_idx"]}__replay_{row["replay_name"]}__window_{row["window_idx"]}.png'
                path = safe_path(folder / "png_files", filename)
                assets = {"encoded_png": relative(self.root, path), "metadata": relative(self.root, metadata)}
                terrain = safe_path(folder / "map_png_files", str(row.get("map_name", "")) + ".png")
                if terrain.is_file():
                    assets["terrain_png"] = relative(self.root, terrain)
                yield Sample(sample_id=stable_id(self.dataset, row["replay_name"], row["window_idx"]),
                             dataset=self.dataset, game=self.game, assets=assets, modalities=list(self.declared_modalities),
                             native_clock={"unit": "windows", "frames_per_window": 255, "approx_seconds_per_window": 10,
                                           "window_idx": int(row["window_idx"]), "timing": "summary_window"},
                             source={"url": "https://github.com/inouye-lab/starcraftimage"},
                             lineage={"replay_name": row["replay_name"], "window_idx": row["window_idx"]},
                             metadata={"native_metadata": row, "representation": "packed_unit_bags", "rgb_gameplay": False})

    def read_window(self, sample, range):
        # One sample is already a summary window. Subdividing the summary cannot
        # recover native gameplay frames or subwindow trajectories.
        if range.unit != "frames" or range.start != 0 or (range.end is not None and range.end != 1):
            raise ValueError("StarCraftImage reads one summary item with TimeRange(0,1,'frames'); no RGB timeline exists")
        image_module = require_dependency("PIL.Image")
        np = require_dependency("numpy")
        with image_module.open(self.asset(sample, "encoded_png")) as image:
            if image.mode != "RGB" or image.width * image.height > MAX_SPATIAL_BYTES // 6:
                raise ValueError("Packed PNG must be bounded 8-bit RGB storage, not a color-converted image")
            encoded_bytes = image.width * image.height * 3
            spatial = decode_unit_bags(np.asarray(image))
        if "terrain_png" in sample.assets:
            if 2 * encoded_bytes + sum(array.nbytes for array in spatial.values()) + 3 * 64 * 64 * 3 + 4096 > MAX_SPATIAL_BYTES:
                raise ValueError("Spatial bands and terrain exceed aggregate memory budget")
            with image_module.open(self.asset(sample, "terrain_png")) as image:
                if image.mode != "RGB" or image.size != (64, 64):
                    raise ValueError("Terrain PNG must have native 64x64 RGB storage")
                terrain = np.asarray(image)
                spatial["terrain"] = {"pathing_grid": terrain[..., 0] != 0,
                                      "placement_grid": terrain[..., 1] != 0, "terrain_height": terrain[..., 2].copy()}
        return Window(sample_id=sample.sample_id, spatial=spatial,
                      states=[sample.metadata["native_metadata"]],
                      metadata={"rgb_gameplay": False, "timebase": sample.native_clock,
                                "unit_ids": "native channel IDs; no unverified name conversion"})

    def validate(self, sample):
        findings = super().validate(sample)
        if findings:
            return findings
        try:
            from ..core import TimeRange
            self.read_window(sample, TimeRange(0, 1, "frames"))
        except (ImportError, MissingDependency) as error:
            findings.append(Finding("missing_dependency", str(error), "warning", sample.sample_id))
        except (ValueError, OSError, KeyError) as error:
            findings.append(Finding("invalid_spatial_png", str(error), sample_id=sample.sample_id))
        return findings
