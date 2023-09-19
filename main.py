import argparse # for command line arguments


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Add arguments
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')
    parser.add_argument('--sum', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='sum the integers (default: find the max)')

    # Parse arguments
    args = parser.parse_args()
    print(args.accumulate(args.integers))
    
if __name__ == '__main__':
    main()