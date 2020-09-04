# Update documentation pharmpy.github.io before release

cp -RT dist/docs/ ../pharmpy.github.io/latest/
cd ../pharmpy.github.io
git add -A
git commit -m "Documentation update"
git push
cd ../pharmpy
