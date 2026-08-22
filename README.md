# CSE 6740: Computational Data Analysis

The permanent public lecture-material hub for CSE 6740 at Georgia Tech, organized by semester.

**Website:** <https://gt-koala.github.io/Gatech-CSE6740/>

**Repository:** <https://github.com/GT-KOALA/Gatech-CSE6740>

## Repository structure

- `_lectures/<semester>/` — one Markdown page per lecture, grouped by offering
- `_data/semesters.yml` — semester metadata and archive ordering
- `_layouts/` — shared page templates
- `semesters/<semester>/index.md` — one schedule page per offering
- `assets/` — shared styles and semester-specific lecture media
- `CONTRIBUTING.md` — pull-request instructions for assigned student work

## Preview locally

GitHub Pages builds this repository with Jekyll. With Ruby and Bundler installed:

```bash
bundle install
bundle exec jekyll serve --baseurl ""
```

Then open <http://localhost:4000>. Publishing is automatic after a change is merged into `main`.

## Add a new semester

1. Add the offering to `_data/semesters.yml`.
2. Create `_lectures/<semester>/` and add that semester's lecture Markdown files.
3. Add a matching defaults block in `_config.yml` to set its label, URL, and lecture permalinks.
4. Create `semesters/<semester>/index.md` by copying the previous semester schedule and changing its `semester` value.
5. Update `current_semester` and the homepage link when the new offering becomes current.

Keep shared layouts and styles at the repository root. Keep offering-specific images and downloadable materials under `assets/semesters/<semester>/`.

## Content boundary

This repository is for public lecture materials only. Homework, exams, grades, solutions, and private student information must not be committed.
