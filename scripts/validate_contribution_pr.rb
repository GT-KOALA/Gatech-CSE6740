#!/usr/bin/env ruby
# frozen_string_literal: true

require "yaml"
require "English"

ROOT = File.expand_path("..", __dir__)
POLICY_PATH = File.join(ROOT, ".github/student-contribution-policy.yml")
LOCKED_FIELDS = %w[semester lecture_number slot slot_order role role_label tab_title assignee issue].freeze
UNSAFE_PATTERNS = {
  "Liquid tag" => /\{%|\{\{/,
  "script element" => /<\s*script/i,
  "embedded frame or object" => /<\s*(iframe|object|embed)/i,
  "inline event handler" => /\son[a-z]+\s*=/i,
  "javascript URL" => /javascript\s*:/i
}.freeze

def fail_check(message)
  warn "Contribution policy failed: #{message}"
  exit 1
end

def front_matter(content, source)
  match = content.match(/\A---\s*\n(.*?)\n---\s*\n/m)
  fail_check("#{source} is missing YAML front matter") unless match
  YAML.safe_load(match[1], permitted_classes: [], aliases: false) || {}
rescue Psych::SyntaxError => e
  fail_check("#{source} has invalid YAML: #{e.message}")
end

def git(*args)
  output = IO.popen(["git", *args], err: [:child, :out], &:read)
  fail_check("git #{args.join(' ')} failed") unless $CHILD_STATUS.success?
  output
end

def validate_structure(policy)
  required = policy.fetch("required_slots")
  lecture_files = Dir.glob(File.join(ROOT, "_lectures/fall-2026/*.md"))
  contribution_files = Dir.glob(File.join(ROOT, "_contributions/fall-2026/**/*.md"))
  fail_check("expected 24 Fall 2026 lectures, found #{lecture_files.length}") unless lecture_files.length == 24
  fail_check("expected 96 contribution files, found #{contribution_files.length}") unless contribution_files.length == 96

  grouped = contribution_files.group_by { |file| front_matter(File.read(file), file).fetch("lecture_number") }
  (1..24).each do |number|
    lecture = format("%02d", number)
    files = grouped.fetch(lecture, [])
    fail_check("lecture #{lecture} must have exactly four contribution slots") unless files.length == 4
    slots = files.to_h do |file|
      metadata = front_matter(File.read(file), file)
      [metadata.fetch("slot"), metadata.fetch("role")]
    end
    fail_check("lecture #{lecture} has incorrect slots or roles") unless slots == required
  end
end

policy = YAML.safe_load(File.read(POLICY_PATH), permitted_classes: [], aliases: false)
validate_structure(policy)

actor = ENV.fetch("GITHUB_ACTOR", "").delete_prefix("@").downcase
base_sha = ENV["BASE_SHA"]
head_sha = ENV["HEAD_SHA"]
exit 0 if base_sha.to_s.empty? || head_sha.to_s.empty?
exit 0 if policy.fetch("staff", []).map(&:downcase).include?(actor)

changed = git("diff", "--name-only", "#{base_sha}...#{head_sha}").lines.map(&:strip).reject(&:empty?)
limit = policy.fetch("maximum_changed_files_per_student_pr", 1)
fail_check("student pull requests must change exactly one assigned section") unless changed.length.between?(1, limit)

file = changed.first
allowed_prefix = "#{policy.fetch('allowed_collection')}/"
fail_check("students may only edit their assigned file under #{allowed_prefix}") unless file.start_with?(allowed_prefix) && file.end_with?(".md")

base_content = git("show", "#{base_sha}:#{file}")
head_content = File.read(File.join(ROOT, file))
base_metadata = front_matter(base_content, "base version of #{file}")
head_metadata = front_matter(head_content, file)

assignee = base_metadata.fetch("assignee", "unassigned").to_s.delete_prefix("@").downcase
fail_check("#{file} is assigned to @#{assignee}, not @#{actor}") unless assignee == actor

LOCKED_FIELDS.each do |field|
  fail_check("students may not change the locked '#{field}' field") unless base_metadata[field] == head_metadata[field]
end

issue = base_metadata["issue"]
fail_check("the course team must set an issue number before student work begins") unless issue.is_a?(Integer) && issue.positive?
body = ENV.fetch("PR_BODY", "")
unless body.match?(/(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#0*#{issue}\b/i)
  fail_check("the pull request body must contain 'Closes ##{issue}'")
end

fail_check("the assigned section exceeds 100 KB") if head_content.bytesize > 100_000
UNSAFE_PATTERNS.each do |label, pattern|
  fail_check("#{file} contains a prohibited #{label}") if head_content.match?(pattern)
end

puts "Contribution policy passed for @#{actor}: #{file}"
