import json
from any2sim import render_pipeline as RP
from any2sim.tools.synth_render_accelerated import reusable


def test_reuse_only_valid_same_appearance(tmp_path):
    ep = tmp_path / 'synthesis/episodes/ep_0'
    rd = tmp_path / 'synthesis/renders/ep_0'
    ep.mkdir(parents=True); rd.mkdir(parents=True)
    for f in ['variant.yaml', 'commanded.npz', 'achieved.npz']:
        (ep / f).write_bytes(b'unchanged')
    snapshot = {'fingerprint': 'frozen'}
    cfg = {'cameras': ['head'], 'samples': 64, 'resume': True}
    (rd / 'head.mp4').write_bytes(b'video')
    d = {'complete': True, 'preview': False, 'render_config': cfg,
         'baseline_fingerprint': 'frozen', 'render_fingerprint': RP.episode_fingerprint(ep, snapshot, cfg)}
    (rd / 'render.json').write_text(json.dumps(d))
    requested = {**cfg, 'baseline': snapshot}
    assert reusable(tmp_path, 'ep_0', requested)
    assert not reusable(tmp_path, 'ep_0', {**requested, 'samples': 32})
    assert not reusable(tmp_path, 'ep_0', {**requested, 'resume': False})
    assert not reusable(tmp_path, 'ep_0', {**requested, 'baseline': {'fingerprint': 'changed'}})
    (ep / 'achieved.npz').write_bytes(b'changed')
    assert not reusable(tmp_path, 'ep_0', requested)
