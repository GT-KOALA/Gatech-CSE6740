# CSE 6740: Computational Data Analysis · Fall 2026

The public, collaborative lecture hub for CSE 6740 at Georgia Tech.

**Website:** <https://gt-koala.github.io/CSE6740-Fall2026/>

## Repository structure

- `_lectures/` — one Markdown page per lecture
- `_layouts/` — shared page templates
- `assets/` — styles and lecture media
- `schedule.md` — generated lecture index
- `contribute.md` — student-facing contribution workflow
- `CONTRIBUTING.md` — detailed GitHub instructions

## Preview locally

GitHub Pages builds this repository with Jekyll. With Ruby and Bundler installed:

```bash
bundle install
bundle exec jekyll serve --baseurl ""
```

Then open <http://localhost:4000>. Publishing is automatic after a change is merged into `main`.

## Content boundary

This repository is for public lecture materials only. Homework, exams, grades, solutions, and private student information must not be committed.

