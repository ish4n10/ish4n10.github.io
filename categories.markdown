---
layout: default
title: Categories
permalink: /categories/
---

<section class="listing-page">
  <h1>Categories</h1>
  <p class="listing-subtitle">Browse posts by topic.</p>

  <div class="group-list">
    {%- assign sorted_tags = site.tags | sort -%}
    {%- for tag in sorted_tags -%}
      {%- assign tag_name = tag[0] -%}
      {%- assign tag_posts = tag[1] -%}
      <details class="group-card">
        <summary>
          <span class="group-title">{{ tag_name }}</span>
          <span class="group-count">{{ tag_posts | size }} post{% if tag_posts.size != 1 %}s{% endif %}</span>
        </summary>
        <ul>
          {%- for post in tag_posts -%}
            <li>
              <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
              <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %-d, %Y" }}</time>
            </li>
          {%- endfor -%}
        </ul>
      </details>
    {%- endfor -%}
  </div>
</section>
