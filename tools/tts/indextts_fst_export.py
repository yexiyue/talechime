"""Development-only export of fixed Linux/Spanish grammars; no product Python."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import zipfile

SPANISH = [
    'Hola, el precio es de veinte euros.', 'El precio es 19,99 € y sube un 5%.',
    'En 2026 viajaré a Madrid.', 'Hoy es 11 de noviembre de 2026.',
    'La fecha es 11/11/2026.', 'El coche S600 alcanza 500 km/h.',
    'Tengo 2 libros y 3,5 litros.', 'Son las 08:30 de la mañana.',
    'Teléfono: +34 612 345678.', 'La temperatura es de -3 °C.',
    'El 21.º participante llegó.', 'Un 25% de descuento.',
    'Alice dice hello y continúa la historia.', '<hola|OW1> 2026',
    '  Hola   mundo.  ', '🙂 2026, fin.',
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tree_identity(package):
    directory = Path(package.__file__).parent
    files = {}
    for path in sorted(directory.rglob('*')):
        if path.is_file() and path.suffix in {'.py', '.tsv', '.txt'}:
            files[path.relative_to(directory).as_posix()] = sha(path.read_bytes())
    return sha(json.dumps(files, sort_keys=True).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='new output directory')
    args = parser.parse_args()
    for package, version in [('WeTextProcessing', '1.3.0'), ('nemo_text_processing', '1.2.0'), ('pynini', '2.1.6.post1')]:
        if importlib.metadata.version(package) != version:
            raise ValueError(f'{package} must be {version}')
    import tn
    import nemo_text_processing
    from tn.chinese.normalizer import Normalizer as Chinese
    from tn.english.normalizer import Normalizer as English
    from nemo_text_processing.text_normalization.normalize import Normalizer as Nemo
    args.output.mkdir(parents=True, exist_ok=False)
    processors = {
        'linux/zh': Chinese(remove_interjections=False, remove_erhua=False),
        'linux/en': English(),
        'es': Nemo(input_case='cased', lang='es'),
    }
    for name, processor in processors.items():
        directory = args.output / name
        directory.mkdir(parents=True)
        if name == 'es':
            tagger, verbalizer = processor.tagger.fst, processor.verbalizer.fst
        else:
            tagger, verbalizer = processor.tagger, processor.verbalizer
        tagger.write(str(directory/'tagger.fst'))
        verbalizer.write(str(directory/'verbalizer.fst'))
    spanish = dict(nemo_text_processing='1.2.0', pynini='2.1.6.post1', input_case='cased', lang='es',
                   cases=[dict(text=text, output=processors['es'].normalize(text, verbose=False)) for text in SPANISH])
    (args.output/'spanish-oracle.json').write_text(json.dumps(spanish, ensure_ascii=False, indent=2)+'\n')
    manifest = dict(format='indextts-front-fst-v1',
                    WeTextProcessing='1.3.0', nemo_text_processing='1.2.0', pynini='2.1.6.post1',
                    source_tree_sha256={'WeTextProcessing': tree_identity(tn), 'NeMo': tree_identity(nemo_text_processing)},
                    license='Apache-2.0', files={})
    for path in sorted(args.output.rglob('*')):
        if path.is_file():
            manifest['files'][path.relative_to(args.output).as_posix()] = dict(bytes=path.stat().st_size, sha256=sha(path.read_bytes()))
    for package in ('wetextprocessing', 'nemo_text_processing'):
        dist = importlib.metadata.distribution(package)
        license_file = next(path for path in dist.files if path.name == 'LICENSE')
        data = dist.locate_file(license_file).read_bytes()
        filename = 'LICENSE-'+package.upper()
        (args.output/filename).write_bytes(data)
        manifest['files'][filename] = dict(bytes=len(data), sha256=sha(data))
    (args.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)+'\n')
    archive = args.output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target:
        for path in sorted(args.output.rglob('*')):
            if path.is_file():
                info = zipfile.ZipInfo(path.relative_to(args.output).as_posix(), date_time=(1980,1,1,0,0,0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                target.writestr(info, path.read_bytes())
    print(json.dumps(dict(file=str(archive), bytes=archive.stat().st_size, sha256=sha(archive.read_bytes()))))


if __name__ == '__main__':
    main()
