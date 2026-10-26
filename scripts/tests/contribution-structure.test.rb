require 'minitest/autorun'
require 'tmpdir'
require 'fileutils'
require 'open3'
require 'rbconfig'

class ContributionStructureTest < Minitest::Test
  ROOT = File.expand_path('../..', __dir__)

  def setup
    @root = Dir.mktmpdir('contribution-structure')
    %w[scripts .github _lectures _contributions].each do |path|
      FileUtils.cp_r(File.join(ROOT, path), @root)
    end
  end

  def teardown
    FileUtils.remove_entry(@root)
  end

  def validate
    Open3.capture2e({'BASE_SHA' => '', 'HEAD_SHA' => ''}, RbConfig.ruby,
      File.join(@root, 'scripts/validate_contribution_pr.rb'))
  end

  def test_extra_session_page_is_allowed
    output, status = validate
    assert status.success?, output
  end

  def test_missing_original_lecture_is_rejected_even_with_extra_session
    FileUtils.rm(File.join(@root, '_lectures/fall-2026/17-transformers.md'))
    output, status = validate
    refute status.success?
    assert_includes output, 'missing Fall 2026 lecture groups: 17'
  end

  def test_missing_contribution_is_rejected
    FileUtils.rm(Dir.glob(File.join(@root, '_contributions/fall-2026/**/*.md')).first)
    output, status = validate
    refute status.success?
    assert_includes output, 'expected 96 contribution files, found 95'
  end
end
