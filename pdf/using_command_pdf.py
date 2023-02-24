import os
import subprocess


def compress_pdf(input_path, output_path, quality=200):
    # Check if input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError('Input file does not exist.')

    # Define the ghostscript command
    command = ['gs', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4', '-dPDFSETTINGS=/screen',
               f'-dNOPAUSE', '-dQUIET', '-dBATCH', f'-dDownsampleColorImages=true',
               f'-dColorImageResolution={quality}', f'-sOutputFile={output_path}', input_path]

    # Call the ghostscript command
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check if output file exists
    if not os.path.isfile(output_path):
        raise RuntimeError('Failed to compress PDF.')


compress_pdf('/Users/pradeep/Downloads/20230223211800.pdf', '/Users/pradeep/Downloads/output.pdf')