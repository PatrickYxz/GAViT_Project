const sharp=require('/Users/patrick/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp');
const fs=require('fs'); const p='/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2/results/evidence/nwpu_regions_review_20260929/';
(async()=>{
 for(let i=0;i<90;i++) {const id=String(i).padStart(3,'0'); await sharp(p+id+'_main.svg').png().toFile(p+id+'_main.png');}
 for(let start=0;start<90;start+=10){const layers=[];for(let j=0;j<10;j++){const id=String(start+j).padStart(3,'0');layers.push({input:await sharp(p+id+'_main.png').resize(768,308).toBuffer(),top:Math.floor(j/2)*312,left:(j%2)*772});}await sharp({create:{width:1540,height:1560,channels:3,background:'#e5e7eb'}}).composite(layers).png().toFile(p+'contact_'+String(start/10)+'.png');}
 for(const i of [15,27,28,46,54,60,64,68,45,84,87]){const id=String(i).padStart(3,'0');await sharp(p+id+'_memberships.svg').png().toFile(p+id+'_memberships.png');}
 console.log('RENDERED: 90 full panels, 9 contact sheets, 11 membership panels');
})().catch(e=>{console.error(e);process.exit(1)});
