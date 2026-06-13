-- tiers.lua — pandoc filter for the three-tier content model
-- TIER env var: core (executives) | student | full (+ interested reader)
-- Works with pandoc 2.9+ (uses env var, no Meta-ordering tricks)

local tier = os.getenv("TIER") or "full"
local keep_student = (tier ~= "core")
local keep_deep = (tier == "full")

local function has_class(div, name)
  for _, c in ipairs(div.classes) do
    if c == name then return true end
  end
  return false
end

local function is_latex()
  return FORMAT:match("latex") ~= nil
end

-- Boxed aside for PDF output: bold label + indented quote block
local function latex_box(div, label)
  local blocks = {}
  table.insert(blocks, pandoc.Para({ pandoc.Strong({ pandoc.Str(label) }) }))
  for _, b in ipairs(div.content) do table.insert(blocks, b) end
  return pandoc.BlockQuote(blocks)
end

function Div(div)
  if has_class(div, "student") then
    if not keep_student then return {} end
    if is_latex() then return latex_box(div, "Student tier.") end

  elseif has_class(div, "deep") then
    if not keep_deep then return {} end
    if is_latex() then return latex_box(div, "For the interested reader.") end

  elseif has_class(div, "widget") then
    if is_latex() then
      -- PDFs can't be interactive: leave a pointer to the website
      return pandoc.Para({ pandoc.Emph({
        pandoc.Str("\u{2192} Interactive component on the website (Play).") }) })
    end

  elseif has_class(div, "animation") then
    if is_latex() then
      return pandoc.Para({ pandoc.Emph({
        pandoc.Str("\u{2192} Animation on the website / in the recording (Manim).") }) })
    end

  elseif has_class(div, "build") then
    if is_latex() then return latex_box(div, "Build exercise (see the notebook).") end

  elseif has_class(div, "example") then
    if is_latex() then return latex_box(div, "Example.") end
  end

  return div
end
