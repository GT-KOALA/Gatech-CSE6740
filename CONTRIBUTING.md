# Contributing to CSE 6740 Fall 2026

Thank you for helping make the course materials clearer and more useful. Student note-taking assignments are coordinated by the course team; completed materials are submitted through GitHub pull requests and reviewed before publication.

## Before you begin

1. Sign up for a lecture section in the Google Sheet shared by the course team.
2. Wait for the course team to assign you a GitHub issue and one exact Markdown file.
3. Coordinate with the other students assigned to that lecture.
4. Use the assigned issue and file as the complete scope for your pull request.

Each lecture has exactly four independent sections:

- `course-1.md` and `course-2.md` cover core course materials.
- `optional-1.md` and `optional-2.md` cover optional extensions or related topics.

## Make a change

1. Fork this repository and create a descriptive branch such as `lecture-02-kmeans-example`.
2. Edit only the exact file named in your issue, under `_contributions/<semester>/<lecture-number>/`.
3. Do not edit the YAML front matter between the two `---` lines. Begin writing below it.
4. Open a pull request using the template. Its body must contain `Closes #NNN` with your assigned issue number.

The automated contribution guard compares your GitHub username with the assignment stored on `main`. It rejects a student pull request if it changes another file, changes assignment metadata, does not close the assigned issue, introduces unsafe embedded code, or exceeds the size limit.

Small, focused pull requests are easier to review. Do not commit generated site files.

## Course-team assignment procedure

Before a student starts, a course-team member performs two small updates directly through a staff pull request:

1. Create and assign the lecture-material issue.
2. In the student's exact section file, replace `assignee: unassigned` with their GitHub username and replace `issue: null` with the numeric issue number.

For example:

```yaml
assignee: octocat
issue: 123
```

After that assignment is merged into `main`, the student forks the repository and edits only the body of that file. Course-team members review and approve the resulting pull request.

## What belongs here

- Clear summaries and intuitive explanations
- Correctly typeset equations
- Small code examples shown directly in Markdown
- Original or appropriately licensed visuals
- Useful references with stable links
- Corrections to factual, mathematical, or typographical errors

## What does not belong here

- Homework, exam, quiz, grading, or solution content
- Private course communications or student information
- Copyrighted material copied without permission
- New files, uploads, or generated binaries in a student pull request
- Liquid templates, scripts, embedded frames, or executable HTML
- AI-generated content that has not been checked and disclosed

By contributing, you agree that your work can be shared under this repository's license and that the course team may edit it for accuracy, clarity, accessibility, and consistency.
