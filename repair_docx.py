"""Repair Project_Report_Draft.docx: the archive embeds 8 PNG images whose
media parts carry an '.undefined' extension and no [Content_Types] entry,
which makes the package unreadable to Word and python-docx alike. All eight
.undefined parts are PNG (verified via file magic), so a single Default
content-type for the extension makes the package conformant.
"""

import zipfile
import shutil

SRC = "Project_Report_Draft.docx"
DST = "Project_Report_Draft_repaired.docx"

ctype_name = "[Content_Types].xml"
injected = False

with zipfile.ZipFile(SRC, "r") as zin, zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == ctype_name:
            xml = data.decode("utf-8")
            if "Extension=\"undefined\"" in xml:
                print("[SKIP] content type for .undefined already present")
            else:
                xml = xml.replace(
                    "</Types>",
                    '<Default ContentType="image/png" Extension="undefined"/></Types>',
                )
                data = xml.encode("utf-8")
                injected = True
        zout.writestr(item, data)

print(f"[{'OK' if injected else 'NOOP'}] rebuilt {DST}")
if injected:
    shutil.move(DST, SRC)
    print(f"moved over {SRC}")