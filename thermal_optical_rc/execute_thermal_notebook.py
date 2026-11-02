"""Execute the notebook locally and embed text and matplotlib figures."""
import base64
import contextlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

path = Path('thermal_free_decay_comparison.ipynb')
notebook = json.loads(path.read_text(encoding='utf-8'))
namespace = {'__name__': '__main__'}
count = 0
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    count += 1
    outputs = []
    stdout = io.StringIO()

    def flush_text():
        text = stdout.getvalue()
        if text:
            outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': text.splitlines(keepends=True)})
            stdout.seek(0)
            stdout.truncate()

    def capture_show(*args, **kwargs):
        flush_text()
        for number in plt.get_fignums():
            fig = plt.figure(number)
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight')
            outputs.append({'output_type': 'display_data', 'metadata': {}, 'data': {
                'image/png': base64.b64encode(buffer.getvalue()).decode('ascii'),
                'text/plain': f'<Figure size {fig.get_size_inches()[0]:g}x{fig.get_size_inches()[1]:g}>'}})
        plt.close('all')

    plt.show = capture_show
    with contextlib.redirect_stdout(stdout):
        exec(compile(''.join(cell['source']), f'notebook cell {count}', 'exec'), namespace)
    flush_text()
    cell['execution_count'] = count
    cell['outputs'] = outputs
    print(f'Executed cell {count}: {len(outputs)} output(s)')

path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
print('Notebook execution complete:', path)
