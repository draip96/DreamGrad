"""Artifact and exact-step contracts for memory experiment analysis."""

import json
import sys

import pytest

from experiments import analyze_memory


def _write_run(path, *, length=8, episodes=1000, extra_steps=0):
  rows = length + 2
  steps = episodes * rows + extra_steps
  (path / 'config.yaml').write_text(f'run:\n  steps: {steps}\n')
  scores = [
      {'step': (index + 1) * rows, 'episode/score': 1.0}
      for index in range(episodes)]
  metrics = [
      {'step': (index + 1) * rows, 'episode/length': float(rows)}
      for index in range(episodes)]
  (path / 'scores.jsonl').write_text(
      ''.join(json.dumps(row) + '\n' for row in scores))
  (path / 'metrics.jsonl').write_text(
      ''.join(json.dumps(row) + '\n' for row in metrics))


def test_toy_analysis_accepts_exact_complete_episode_budget(
    tmp_path, monkeypatch, capsys):
  _write_run(tmp_path)
  monkeypatch.setattr(sys, 'argv', [
      'analyze_memory.py', str(tmp_path), '--kind', 'toy', '--length', '8'])
  with pytest.raises(SystemExit) as info:
    analyze_memory.main()
  assert info.value.code == 0
  result = json.loads(capsys.readouterr().out)
  assert result['episodes'] == 1000
  assert result['expected_steps'] == 10_000
  assert result['final_score_step'] == 10_000
  assert result['passed'] is True


def test_toy_analysis_rejects_partial_final_episode(tmp_path, monkeypatch):
  _write_run(tmp_path, extra_steps=10)
  monkeypatch.setattr(sys, 'argv', [
      'analyze_memory.py', str(tmp_path), '--kind', 'toy', '--length', '8'])
  with pytest.raises(RuntimeError, match='partial final episode'):
    analyze_memory.main()
