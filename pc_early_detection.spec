# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, copy_metadata

# Collect all needed files and metadata for these libraries
_s_datas, _s_binaries, _s_hidden = collect_all('streamlit')
_sk_datas, _sk_binaries, _sk_hidden = collect_all('sklearn')
_pd_datas, _pd_binaries, _pd_hidden = collect_all('pandas')
_np_datas, _np_binaries, _np_hidden = collect_all('numpy')
_jb_datas, _jb_binaries, _jb_hidden = collect_all('joblib')
_wv_datas, _wv_binaries, _wv_hidden = collect_all('webview')
_st_meta = copy_metadata('streamlit')

# App data files to ship next to the executable
_app_datas = [
    ('pc_early_detection.py', '.'),
    ('model.joblib', '.'),
    ('columns.json', '.'),
    ('metrics.txt', '.'),
]

a = Analysis(
    ['native_app.py'],
    pathex=[],
    binaries=_s_binaries + _sk_binaries + _pd_binaries + _np_binaries + _jb_binaries + _wv_binaries,
    datas=_s_datas + _sk_datas + _pd_datas + _np_datas + _jb_datas + _st_meta + _app_datas + _wv_datas,
    hiddenimports=_s_hidden + _sk_hidden + _pd_hidden + _np_hidden + _jb_hidden + _wv_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='pc_early_detection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
