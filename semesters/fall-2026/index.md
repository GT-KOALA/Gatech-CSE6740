---
layout: default
title: Lectures
permalink: /fall-2026/
semester: fall-2026
---
<section class="page-intro">
  <p class="eyebrow">Fall 2026 · Tentative schedule</p>
  <h1>Lecture index</h1>
  <p>Topics may shift as the semester develops. Open a lecture to find its notes, slides, examples, code, and references.</p>
</section>

<section class="lecture-index">
{% assign semester_lectures = site.lectures | where: "semester", page.semester %}
{% assign sorted_lectures = semester_lectures | sort: "date" %}
{% for lecture in sorted_lectures %}
  <a class="lecture-row" href="{{ lecture.url | relative_url }}">
    <span class="lecture-number">{{ lecture.number }}</span>
    <span class="lecture-date">{{ lecture.date | date: "%b %-d" }}</span>
    <span class="lecture-title">{{ lecture.title }}</span>
    <span class="lecture-team">Materials</span>
    <span class="arrow">↗</span>
  </a>
{% endfor %}
</section>

<p class="schedule-note">No-class dates and assessment information are communicated separately.</p>
