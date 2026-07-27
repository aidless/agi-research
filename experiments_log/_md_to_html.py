import markdown
import os
with open(r'E:\agi-research\thesis_draft_v1.0.md', 'r', encoding='utf-8') as f:
    text = f.read()
html = markdown.markdown(text, extensions=['tables', 'fenced_code', 'toc', 'attr_list'])
header = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Archimedes Project - Thesis Draft v1.0</title>
<style>
body { font-family: "Times New Roman", serif; max-width: 820px; margin: 40px auto; line-height: 1.6; padding: 0 20px; color: #222; }
h1 { font-size: 2em; border-bottom: 2px solid #333; padding-bottom: 8px; }
h2 { font-size: 1.6em; margin-top: 1.5em; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
h3 { font-size: 1.3em; margin-top: 1.2em; color: #444; }
h4 { font-size: 1.1em; color: #555; }
code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-family: "Consolas", monospace; }
pre { background: #f8f8f8; padding: 12px; border-radius: 5px; overflow-x: auto; border: 1px solid #ddd; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 16px 0; }
th, td { border: 1px solid #ccc; padding: 6px 12px; text-align: left; }
th { background: #eee; }
blockquote { border-left: 4px solid #ddd; margin: 16px 0; padding: 0 16px; color: #555; }
hr { border: 0; border-top: 1px dashed #999; margin: 32px 0; }
</style>
</head>
<body>
'''
footer = '''
</body>
</html>
'''
out = header + html + footer
out_path = r'E:\agi-research\thesis_draft_v1.0.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(out)
print(f'HTML written: {out_path}')
print(f'Size: {os.path.getsize(out_path)} bytes')
print(f'Sections: {text.count(chr(10) + "## ")}')
print(f'Lines: {len(text.splitlines())}')
