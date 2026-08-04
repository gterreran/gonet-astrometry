from gonet_astrometry.cli import build_parser, main


def test_parser_has_expected_program_name() -> None:
    assert build_parser().prog == "gonet-astrometry"


def test_main_accepts_no_arguments() -> None:
    assert main([]) == 0
