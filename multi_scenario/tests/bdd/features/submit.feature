Feature: Submit page workflow
  As a researcher running multi-agent experiments
  I want the Submit page to walk me through pick → edit → save → preflight → submit
  So I get clear feedback before any compute is wasted

  Background:
    Given an experiments directory with a valid discovery baseline config

  Scenario: Submit a local run end-to-end
    Given the local submit target is selected
    When I pick discovery / baseline / seed0.yaml
    And I run preflight
    And I click Submit with a stubbed LocalRunner
    Then the submission status is "done"

  Scenario: OVH preflight cascades when configs/ovh.yaml is missing
    Given configs/ovh.yaml is absent
    When I pick discovery / baseline / seed0.yaml
    And I switch the submit target to ovh
    And I run preflight
    Then the "OVH config valid" row is FAIL
    And every other OVH probe row stays IDLE
