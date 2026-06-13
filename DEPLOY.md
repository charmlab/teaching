# Deploying The AI Learning Map

Target: **teaching.amirhkarimi.com** (the `CNAME` file says this — if you actually
meant a different domain like `amirhq.com`, edit `CNAME` accordingly).

## 1. GitHub Pages (same setup as your main site)

```bash
cd "2026.06 - new cto ai"
git init && git add index.html module.html data.js styles.css CNAME
git commit -m "AI Learning Map v1"
# create a repo, e.g. under your org:
# https://github.com/charmlab/teaching  (or amirhk/teaching)
git remote add origin https://github.com/charmlab/teaching.git
git push -u origin main
```

Then in the repo: **Settings → Pages → Source: main branch, / (root)**.
GitHub reads `CNAME` automatically and provisions HTTPS (enable "Enforce HTTPS"
once the cert is issued, ~15 min).

## 2. DNS (wherever amirhkarimi.com is registered)

Add one record:

| Type  | Host       | Value                  |
|-------|------------|------------------------|
| CNAME | `teaching` | `<github-user-or-org>.github.io` |

e.g. `charmlab.github.io` if the repo lives under the charmlab org
(same place as your main site's repo).

## 3. Google Analytics

- The site reuses your existing GA4 property: **G-PLH5YY7PQS** (set in `data.js`,
  top of file). Subdomain traffic appears under the same property — filter by
  the `hostname` dimension (`teaching.amirhkarimi.com`) to separate it from the
  main site. If you'd rather keep teaching analytics fully separate, create a
  new GA4 data stream/property and swap `GA_ID` in `data.js`.
- In GA4, check **Admin → Data streams → Enhanced measurement** is ON: this
  gives you scroll depth, outbound link clicks, and engagement time for free —
  i.e. "where people are spending time."
- Custom events already wired (visible in GA4 under Reports → Engagement → Events
  after ~24h, or instantly in Admin → DebugView):

| Event               | Fired when                                  | Params |
|---------------------|---------------------------------------------|--------|
| `audience_toggle`   | Student/Executive switch                    | `audience` |
| `view_toggle`       | Skill tree / Your path switch               | `view` |
| `module_popup`      | A node or path card is clicked              | `module_id`, `audience` |
| `ingredient_select` | Listen/Play/Build chosen in the popup       | `module_id`, `ingredient` |
| `resource_click`    | An external resource link is clicked        | `module_id`, `resource` |
| `module_view`       | A module page is opened                     | `module_id`, `mode`, `reference` |
| `minimap_travel`    | Navigation via the mini skill tree          | `from_module`, `to_module` |
| `quiz_complete`     | Placement quiz finished ("Show me" clicked) | `audience`, `start_module` |

Tip: register `module_id`, `audience`, and `ingredient` as **custom dimensions**
(Admin → Custom definitions) so they show up in standard reports.

## 4. Sanity checklist after first deploy

- [ ] https://teaching.amirhkarimi.com loads, map renders, no console errors
- [ ] Clicking a foundations node shows reference links (not Listen/Play/Build)
- [ ] GA4 DebugView shows events while you click around
- [ ] Quiz appears in a private/incognito window (first-visit logic)
