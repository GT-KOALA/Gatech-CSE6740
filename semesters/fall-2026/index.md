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

{% assign term = site.data.semesters | where: "id", page.semester | first %}
<section class="course-facts" aria-label="Fall 2026 course details">
  <div><span>When</span><strong>Mon & Wed<br>12:30–1:45 PM</strong></div>
  <div><span>Where</span><strong>{{ term.location }}</strong></div>
  <div><span>Instructor</span><strong>{{ term.instructor }}</strong></div>
  <div><span>Drop-in hours</span><strong>{{ term.instructor_hours }}</strong></div>
</section>

<section class="lecture-index">
{% assign semester_lectures = site.lectures | where: "semester", page.semester %}
{% for session in site.data.fall_2026_schedule %}
  {% assign lecture = nil %}
  {% if session.lecture %}
    {% assign lecture = semester_lectures | where: "slug", session.lecture | first %}
  {% endif %}
  {% if lecture %}
  <a class="lecture-row" href="{{ lecture.url | relative_url }}">
    <span class="lecture-number">{{ lecture.number }}</span>
    <span class="lecture-date">{{ session.date | date: "%b %-d" }}</span>
    <span class="lecture-title">{{ lecture.title }}</span>
    <span class="lecture-team">Materials</span>
    <span class="arrow">↗</span>
  </a>
  {% else %}
  <div class="lecture-row">
    <span class="lecture-number"></span>
    <span class="lecture-date">{{ session.date | date: "%b %-d" }}</span>
    <span class="lecture-title">{{ session.title }}</span>
  </div>
  {% endif %}
{% endfor %}
</section>

<p class="schedule-note">Blank schedule entries indicate topics to be announced. Assessment information is communicated separately.</p>

<section class="intro-grid teaching-team">
  <div>
    <p class="eyebrow">Teaching team</p>
    <h2>Questions and drop-in hours</h2>
  </div>
  <div class="prose">
    {% for ta in term.teaching_assistants %}
    <p><strong>{{ ta.name }}</strong><br>{{ ta.hours }}</p>
    {% endfor %}
  </div>
</section>
