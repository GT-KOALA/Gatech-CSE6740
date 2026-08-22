---
layout: default
title: Lectures
permalink: /schedule/
---
<section class="page-intro">
  <p class="eyebrow">Fall 2026 · Tentative schedule</p>
  <h1>Lecture index</h1>
  <p>Topics may shift as the semester develops. Open a lecture to find notes, materials, and its four-person student team.</p>
</section>

<section class="lecture-index">
{% assign sorted_lectures = site.lectures | sort: "date" %}
{% for lecture in sorted_lectures %}
  <a class="lecture-row" href="{{ lecture.url | relative_url }}">
    <span class="lecture-number">{{ lecture.number }}</span>
    <span class="lecture-date">{{ lecture.date | date: "%b %-d" }}</span>
    <span class="lecture-title">{{ lecture.title }}</span>
    <span class="lecture-team">4 collaborators</span>
    <span class="arrow">↗</span>
  </a>
{% endfor %}
</section>

<p class="schedule-note">No-class dates and assessment information are communicated separately.</p>

