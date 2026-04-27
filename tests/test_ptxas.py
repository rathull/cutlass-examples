from cutlass_examples.common.ptxas import parse_ptxas_log, render_ptxas_html


SAMPLE_PTXAS_LOG = """
ptxas info    : Compiling entry function '_Z9matmul_v0PK13__nv_bfloat16S1_PS_iii' for 'sm_80'
ptxas info    : Function properties for _Z9matmul_v0PK13__nv_bfloat16S1_PS_iii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 64 registers, 1024 bytes smem, 360 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compiling entry function '_Z9matmul_v1PK13__nv_bfloat16S1_PS_iii' for 'sm_80'
ptxas info    : Function properties for _Z9matmul_v1PK13__nv_bfloat16S1_PS_iii
    16 bytes stack frame, 24 bytes spill stores, 32 bytes spill loads
ptxas info    : Used 80 registers, 2048 bytes smem, 400 bytes cmem[0]
ptxas warning : Registers are spilled to local memory in function '_Z9matmul_v1'
"""


def test_parse_ptxas_log_extracts_resources():
    report = parse_ptxas_log(SAMPLE_PTXAS_LOG)

    assert len(report.records) == 2
    first = report.records[0]
    assert first.arch == "sm_80"
    assert first.registers == 64
    assert first.smem_bytes == 1024
    assert first.stack_frame_bytes == 0
    assert first.total_cmem_bytes == 368
    assert not first.has_spills

    second = report.records[1]
    assert second.registers == 80
    assert second.stack_frame_bytes == 16
    assert second.spill_stores_bytes == 24
    assert second.spill_loads_bytes == 32
    assert second.total_spill_bytes == 56
    assert second.has_spills
    assert report.warnings


def test_render_ptxas_html_contains_core_columns():
    report = parse_ptxas_log(SAMPLE_PTXAS_LOG)
    html = render_ptxas_html(report)

    assert "Registers/thread" in html
    assert "SMEM bytes" in html
    assert "_Z9matmul_v0" in html
