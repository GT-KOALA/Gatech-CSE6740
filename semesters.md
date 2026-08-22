---
layout: default
title: Semesters
permalink: /semesters/
---
<section class="page-intro">
  <p class="eyebrow">Course archive</p>
  <h1>Semesters</h1>
  <p>Browse schedules and lecture materials from each offering of CSE 6740.</p>
</section>

<section class="lecture-index">
{% for semester in site.data.semesters %}
  <a class="lecture-row" href="{{ semester.url | relative_url }}">
    <span class="lecture-number">{% if semester.status == "current" %}Now{% else %}Past{% endif %}</span>
    <span class="lecture-date">{{ semester.dates }}</span>
    <span class="lecture-title">{{ semester.label }}</span>
    <span class="lecture-team">{{ semester.instructor }}</span>
    <span class="arrow">↗</span>
  </a>
{% endfor %}
</section>

