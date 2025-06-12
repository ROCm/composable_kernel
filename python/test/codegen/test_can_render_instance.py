import unittest

class TestCanRenderInstance(unittest.TestCase):
    def test_can_render_instance(self):
        from genck.ops.ck_tile.gemm.instance import GEMM
        from genck.ops.ck_tile.gemm.render import render

        test_instance = GEMM(
            layout_a="Row",
            layout_b="Col",
            layout_c="Row",
            datatype_a="BF16",
            datatype_b="BF16",
            datatype_c="BF16",
            tile_m=256,
            tile_n=256,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            m_is_padded="false",
            n_is_padded="false",
            k_is_padded="false",
            pipeline="CompV3",
            scheduler="Intrawave",
            epilogue="Default",
        )

        rendered_instance = render(test_instance)
        self.assertIn("ck_tile_gemm_universal", rendered_instance)
