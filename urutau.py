import argparse
import textwrap

import method_runner


def parse_args():
    """Function to handle building and parsing of command line arguments"""
    parser = argparse.ArgumentParser(formatter_class=SubcommandHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Hybrid Modal Synthesis Method Runner
                                     ------------------------------------
                                        Required arguments:
                                        --phikyz <path>
                                        --fmk <path>
                                        --string <name>
                                        --tf <dur>
                                        --xp <pos>
                                     ------------------------------------
                                     '''))
    subparsers = parser.add_subparsers(title="commands", dest='command', metavar='')
    plot_parser = subparsers.add_parser('plot', help="plot everything")
    plot_parser.add_argument('--fft', help="plot fft", action="store_true")

    run_parser = subparsers.add_parser('run', help="Hybrid Modal Synthesis Method Runner",
                                       formatter_class=argparse.RawDescriptionHelpFormatter,
                                       description=textwrap.dedent('''\
                                        Hybrid Modal Synthesis Method Runner
                                        ------------------------------------
                                           Required arguments:
                                           --phikyz <path>
                                           --fmk <path>
                                           --string <name>
                                           --tf <dur>
                                           --xp <pos>
                                        ------------------------------------
                                        '''))

    body_group = run_parser.add_argument_group('body')
    body_group.add_argument("--phikyz", help="Path of the PHIKYZ file at the bridge (REQUIRED)",
                            required=True, metavar="<path>")
    body_group.add_argument("--phikyzp", help="Path of the PHIKYZ file at the peg",
                            metavar="<path>", default=None)
    body_group.add_argument("--fmk", help="Path of the FMK file (REQUIRED)", required=True, metavar="<path>")
    body_group.add_argument("--nb", help="body modes (default: max)", type=int, metavar="<num>", default=-1)
    xsi_group = body_group.add_mutually_exclusive_group()
    xsi_group.add_argument("--xsi-fixed", help="fixed xsi for all modes (default: %(default)s)",
                           type=float, metavar="<val>", default=0.1)
    xsi_group.add_argument("--xsi-path", help="Path of the XSI file", metavar="<path>", default=None)

    string_group = run_parser.add_argument_group('string')
    string_group.add_argument("--string", help="string name (REQUIRED)", required=True, metavar="<name>")
    string_group.add_argument("--string-len", help="string length (meters) (default: %(default)s)", type=float,
                              default=0.65, metavar="<len>")
    string_group.add_argument("--string-fmax", help="max frequency of string modes (Hz) (default: %(default)s)",
                              type=float, default=10000, metavar="<len>")
    string_group.add_argument("--string-f0", help="fundamental frequency (Hz) (default: string default f0)",
                              type=float, default=-1, metavar="<freq>")

    pluck_group = run_parser.add_argument_group('pluck', description="ramp function")
    pluck_group.add_argument("--xp", help="pluck position (meters)(0 ref: peg)(REQUIRED)",
                             required=True, metavar="<pos>", type=float)
    pluck_group.add_argument("--pluck-ti", help="starting time of the ramp (seconds)(default: %(default)s)",
                             default=0.001, metavar="<time>", type=float)
    pluck_group.add_argument("--pluck-dp", help="ramp length (seconds)(default: %(default)s)",
                             default=0.008, metavar="<len>", type=float)
    pluck_group.add_argument("--pluck-F0", help="height of the ramp (N)(default: %(default)s)",
                             default=10, metavar="<val>", type=float)
    pluck_group.add_argument("--pluck-gamma", help="pluck angle (radians)(default: pi/2)",
                             default=1.57079632679, metavar="<rad>", type=float)
    simulation_group = run_parser.add_argument_group('simulation')
    simulation_group.add_argument("--tf",
                                  help="Duration of the simulation (seconds) (REQUIRED)",
                                  type=float, required=True, metavar="<dur>")
    simulation_group.add_argument("--method",
                                  help="simulation method: 'cfc' or 'fft' (default: %(default)s)",
                                  choices=['cfc', 'fft'], default="cfc", metavar="<method>")
    simulation_group.add_argument("--pol", help="the number of polarizations: 1 or 2 (default: %(default)s)",
                                  choices=[1, 2], type=int, default=2, metavar="<num>")
    simulation_group.add_argument("--damp",
                                  help="damping method: 'woodhouse' or 'paiva' (default: %(default)s)",
                                  choices=['paiva', 'woodhouse'], default="paiva", metavar="<method>")
    simulation_group.add_argument("--fs",
                                  help="sample frequency (default: %(default)s)",
                                  default=44100, type=int, metavar="<freq>")

    output_group = run_parser.add_argument_group('output')
    output_group.add_argument("-v", "--verbose", action="store_true", help="print progress percentage for cfc")
    output_group.add_argument("--fft", action="store_true", help="generate pdf with graphics")
    output_group.add_argument("--fft-window", type=float, help="size of the fft window (ms)(default: full signal)",
                              default=-1, metavar="<dur>")
    output_group.add_argument("--pluckingpoint", action="store_true", help="generate files for plucking point when cfc")
    output_group.add_argument("--displ", action="store_true", help="generate files for displacement when cfc")
    output_group.add_argument("--vel", action="store_true", help="generate files for velocity when cfc")
    output_group.add_argument("--acc-no", action="store_true", help="does not generate files for acceleration")
    mp3_group = output_group.add_mutually_exclusive_group()
    mp3_group.add_argument("--mp3", action="store_true", help="generate mp3 file")
    mp3_group.add_argument("--mp3-only", action="store_true", help="generate only mp3 file (no wav)")

    return parser.parse_args()


def shell():
    args = parse_args()

    print(args)

    if args.command == 'run':
        if args.mp3_only:
            mp3 = True
            wav = False
        elif args.mp3:
            mp3 = True
            wav = True
        else:
            mp3 = False
            wav = True

        if args.acc_no:
            acc = False
        else:
            acc = True

        if args.xsi_path:
            xsi = args.xsi_path
        else:
            xsi = args.xsi_fixed

        method_runner.run(args.phikyz, args.fmk, args.string, args.tf, args.xp,
                          args.method, args.pol, args.damp, args.fs,
                          args.phikyzp, args.nb, xsi,
                          args.string_len, args.string_fmax, args.string_f0,
                          args.pluck_ti, args.pluck_dp, args.pluck_F0, args.pluck_gamma,
                          args.verbose, args.fft, args.fft_window,
                          args.pluckingpoint, args.displ, args.vel, acc, wav, mp3)

    elif args.command == 'plot':
        print('plot')
    else:
        print("Command Unknown!")


class SubcommandHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def _format_action(self, action):
        parts = super(argparse.RawDescriptionHelpFormatter, self)._format_action(action)
        if action.nargs == argparse.PARSER:
            parts = "\n".join(parts.split("\n")[1:])
        return parts


if __name__ == '__main__':
    shell()