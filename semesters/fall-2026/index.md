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
  <div><span>Instructor</span><strong><a href="#teaching-team">{{ term.instructor }}</a></strong></div>
  <div><span>Questions & support</span><strong><a href="#teaching-team">Teaching team & drop-in hours ↓</a></strong></div>
</section>

<nav class="schedule-modules" aria-label="Jump to a learning area">
{% for module in site.data.fall_2026_modules %}
  {% unless module.id == "introduction" %}
  <a href="#{{ module.id }}">
    <span>{{ module.range }}</span>
    <strong>{{ module.title }}</strong>
    <span class="module-jump" aria-hidden="true">↓</span>
  </a>
  {% endunless %}
{% endfor %}
</nav>

<section class="lecture-index" aria-label="Schedule by learning area">
{% assign semester_lectures = site.lectures | where: "semester", page.semester %}
{% assign session_number = 0 %}
{% for module in site.data.fall_2026_modules %}
<section class="schedule-module" id="{{ module.id }}" aria-labelledby="{{ module.id }}-title">
  <header class="schedule-module-heading">
    <div>
      <p class="eyebrow">{{ module.range }}</p>
      <h2 id="{{ module.id }}-title">{{ module.title }}</h2>
    </div>
    {% if module.description %}<p>{{ module.description }}</p>{% endif %}
  </header>
{% assign sessions = site.data.fall_2026_schedule | where: "group", module.id %}
{% for session in sessions %}
  {% assign session_number = session_number | plus: 1 %}
  {% assign lecture = nil %}
  {% if session.lecture %}
    {% assign lecture = semester_lectures | where: "slug", session.lecture | first %}
  {% endif %}
  {% if lecture %}
  <a class="lecture-row" href="{{ lecture.url | relative_url }}">
    <span class="lecture-number">{{ session_number }}</span>
    <span class="lecture-date">{{ session.date | date: "%b %-d" }}</span>
    <span class="lecture-title">{{ lecture.title }}</span>
    <span class="lecture-team">Materials</span>
    <span class="arrow">↗</span>
  </a>
  {% else %}
  <div class="lecture-row">
    <span class="lecture-number">{{ session_number }}</span>
    <span class="lecture-date">{{ session.date | date: "%b %-d" }}</span>
    <span class="lecture-title">{{ session.title }}</span>
  </div>
  {% endif %}
{% endfor %}
</section>
{% endfor %}
</section>

<p class="schedule-note">Blank schedule entries indicate topics to be announced. Assessment information is communicated separately.</p>

<section class="team-section" id="teaching-team" aria-labelledby="team-title">
  <header class="team-heading">
    <p class="eyebrow">Teaching team</p>
    <h2 id="team-title">Meet the team</h2>
    <p>Bring your course questions to any of our weekly drop-in hours.</p>
  </header>
  <div class="team-grid">
    <article class="team-member">
      <img src="{{ term.instructor_photo | relative_url }}" alt="{{ term.instructor }}" width="160" height="160" loading="lazy">
      <p class="team-role">Instructor</p>
      <h3>{{ term.instructor }}</h3>
      <p class="team-hours"><span>Drop-in hours</span>{{ term.instructor_hours }}</p>
    </article>
    {% for ta in term.teaching_assistants %}
    <article class="team-member">
      <img src="{{ ta.photo | relative_url }}" alt="{{ ta.name }}" width="160" height="160" loading="lazy">
      <p class="team-role">Teaching assistant</p>
      <h3>{{ ta.name }}</h3>
      <p class="team-hours"><span>Drop-in hours</span>{{ ta.hours }}</p>
    </article>
    {% endfor %}
  </div>
</section>
